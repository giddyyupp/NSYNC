#@title Import required libraries
import argparse
import itertools
import math
import os
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel, StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import config as cfg

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


#@markdown `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
pretrained_model_name_or_path = cfg.pretrained_model_name_or_path  # "stabilityai/stable-diffusion-2-base" #@param ["stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-base", "CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"] {allow-input: true}

#@markdown `images_path` is a path to directory containing the training images. It could
images_path = cfg.baseline_images_path #@param {type:"string"}

#@title Settings for your newly created concept
#@markdown `what_to_teach`: what is it that you are teaching? `object` enables you to teach the model a new object to be used, `style` allows you to teach the model a new style one can use.
what_to_teach = cfg.what_to_teach #@param ["object", "style"]
#@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `<my-placeholder-token>` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
placeholder_token = cfg.placeholder_token #@param {type:"string"}
#@markdown `initializer_token` is a word that can summarise what your new concept is, to be used as a starting point
initializer_token = cfg.initializer_token #@param {type:"string"}


#@title Setup the prompt templates for training
imagenet_templates_small = cfg.imagenet_templates_small
imagenet_style_templates_small = cfg.imagenet_style_templates_small

#@title Setup the dataset
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        prompt_dir,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.prompt_dir = prompt_dir
        
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        img_name = self.image_paths[i % self.num_images]
        image = Image.open(img_name)
        img_name = os.path.basename(img_name)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        # text = random.choice(self.templates).format(placeholder_string)
        prompt_name = os.path.splitext(img_name)[0] + ".txt"
        prompt_path = os.path.join(self.prompt_dir, prompt_name)
        with open(prompt_path, "r") as f:
            prompt = f.read()
        style_prompt = prompt + f" in the style of {placeholder_string}"

        example['prompt'] = style_prompt

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

# Load model
base = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path, 
    torch_dtype=torch.bfloat16,
    # variant="fp32", 
    safety_checker=None
)
base.disable_xformers_memory_efficient_attention()


XLt1=base.components["text_encoder"]
XLt2=base.components["text_encoder_2"]
XLtok1=base.components["tokenizer"]
XLtok2=base.components["tokenizer_2"]
XLunet=base.components["unet"]
XLvae=base.components['vae']
XLsch=base.components['scheduler']
base.upcast_vae() # vae does not work correctly in 16 bit mode -> force fp32

####################################################################################################
# Add the placeholder token in tokenizer1
num_added_tokens = XLtok1.add_tokens(placeholder_token)
if num_added_tokens == 0:
    raise ValueError(
        f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
        " `placeholder_token` that is not already in the tokenizer."
    )

#@title Get token ids for our placeholder and initializer token. This code block will complain if initializer string is not a single token
# Convert the initializer_token, placeholder_token to ids
token_ids = XLtok1.encode(initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id = token_ids[0]
placeholder_token_id = XLtok1.convert_tokens_to_ids(placeholder_token)

####################################################################################################
# Add the placeholder token in tokenizer2
num_added_tokens = XLtok2.add_tokens(placeholder_token)
if num_added_tokens == 0:
    raise ValueError(
        f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
        " `placeholder_token` that is not already in the tokenizer."
    )

#@title Get token ids for our placeholder and initializer token. This code block will complain if initializer string is not a single token
# Convert the initializer_token, placeholder_token to ids
token_ids = XLtok2.encode(initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id = token_ids[0]
placeholder_token_id = XLtok2.convert_tokens_to_ids(placeholder_token)

XLt1.resize_token_embeddings(len(XLtok1))
token_embeds1 = XLt1.get_input_embeddings().weight.data
token_embeds1[placeholder_token_id] = token_embeds1[initializer_token_id]

XLt2.resize_token_embeddings(len(XLtok2))
token_embeds2 = XLt2.get_input_embeddings().weight.data
token_embeds2[placeholder_token_id] = token_embeds2[initializer_token_id]


def freeze_params(params):
    for param in params:
        param.requires_grad = False

# Freeze vae and unet
freeze_params(XLvae.parameters())
freeze_params(XLunet.parameters())
# Freeze all parameters except for the token embeddings in text encoder
params_to_freeze = itertools.chain(
    XLt1.text_model.encoder.parameters(),
    XLt1.text_model.final_layer_norm.parameters(),
    XLt1.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)
# Freeze all parameters except for the token embeddings in text encoder
params_to_freeze = itertools.chain(
    XLt2.text_model.encoder.parameters(),
    XLt2.text_model.final_layer_norm.parameters(),
    XLt2.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)


train_dataset = TextualInversionDataset(
      data_root=images_path,
      prompt_dir=cfg.baseline_prompt_dir,
      tokenizer=XLtok1,
      size=XLvae.sample_size,
      placeholder_token=placeholder_token,
      repeats=100,
      learnable_property=what_to_teach, # Option selected above between object and style
      center_crop=False,
      set="train",
)

def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

noise_scheduler = XLsch #DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

#@title Setting up all training args
hyperparameters = cfg.hyperparameters_baseline_ti

#@title Training function
logger = get_logger(__name__)

# weight_dtype = torch.float32


def save_progress(text_encoder1, text_encoder2, global_step, placeholder_token_id, accelerator, output_dir):
    logger.info("Saving embeddings")
    save_path1 = os.path.join(output_dir, f"learned_embeds1-step-{global_step}.bin")
    learned_embeds1 = accelerator.unwrap_model(text_encoder1).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds1.detach().cpu()}
    torch.save(learned_embeds_dict, save_path1)

    save_path2 = os.path.join(output_dir, f"learned_embeds2-step-{global_step}.bin")
    learned_embeds2 = accelerator.unwrap_model(text_encoder2).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds2.detach().cpu()}
    torch.save(learned_embeds_dict, save_path2)


# Helper to get the correct time_ids for SDXL
def compute_time_ids(original_size, crops_coords_top_left, target_size, device, dtype):
    # SDXL expects a tensor of shape (batch, 6) containing:
    # (original_height, original_width, crop_top, crop_left, target_height, target_width)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], device=device, dtype=dtype)
    return add_time_ids

def training_function(text_encoder1, text_encoder2, vae, unet):
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    gradient_checkpointing = hyperparameters["gradient_checkpointing"]

    os.makedirs(output_dir, exist_ok = True)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16"
    )

    if gradient_checkpointing:
        text_encoder1.gradient_checkpointing_enable()
        text_encoder2.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    train_dataloader = create_dataloader(train_batch_size)

    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # CRITICAL: Enable gradients for the embedding layers
    text_encoder1.get_input_embeddings().requires_grad_(True)
    text_encoder2.get_input_embeddings().requires_grad_(True)

    # OPTIMIZER FIX: Ensure we are passing a flat list of parameters
    optimizer = torch.optim.AdamW(
        list(text_encoder1.get_input_embeddings().parameters()) + 
        list(text_encoder2.get_input_embeddings().parameters()),
        lr=learning_rate,
    )

    text_encoder1, text_encoder2, optimizer, train_dataloader = accelerator.prepare(
        text_encoder1, text_encoder2, optimizer, train_dataloader
    )

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    unet.train()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # --- SDXL SPECIFIC SETUP ---
    # We assume square training images based on your dataset
    resolution = XLvae.sample_size # Likely 1024 or 512
    # Create fixed time_ids since your dataset resizes everything to square
    # (original_h, original_w, crop_top, crop_left, target_h, target_w)
    default_time_ids = compute_time_ids((resolution, resolution), (0, 0), (resolution, resolution), accelerator.device, weight_dtype)
    
    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder1.train()
        text_encoder2.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder1), accelerator.accumulate(text_encoder2):
                
                # 1. VAE Encoding
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # 2. Noise Generation
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. MANUAL TEXT ENCODING (Replaces base.encode_prompt)
                # We tokenize manually here to ensure we use the learnable embeddings
                
                # Tokenizer 1 (CLIP ViT-L)
                tokens1 = XLtok1(batch['prompt'], padding="max_length", max_length=XLtok1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                # Get hidden states (output[0] is last hidden state, output[1] is pooled - but CLIP L doesn't use pooled in SDXL)
                enc1_output = text_encoder1(tokens1, output_hidden_states=True)
                # Use penultimate layer hidden states for SDXL
                enc1_hidden = enc1_output.hidden_states[-2].to(dtype=weight_dtype)

                # Tokenizer 2 (OpenCLIP ViT-G)
                tokens2 = XLtok2(batch['prompt'], padding="max_length", max_length=XLtok2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                enc2_output = text_encoder2(tokens2, output_hidden_states=True)
                # Use penultimate layer hidden states
                enc2_hidden = enc2_output.hidden_states[-2].to(dtype=weight_dtype)

                # Get pooled output (for time_ids conditioning)
                pooled_embeds = enc2_output.text_embeds.to(dtype=weight_dtype) # Size: [batch, 1280]

                # Concatenate the two text encoders (Feature dimension: 768 + 1280 = 2048)
                prompt_embeds = torch.cat([enc1_hidden, enc2_hidden], dim=-1)

                # 4. PREPARE ADDED CONDITIONS
                # Repeat time_ids for the batch
                add_time_ids = default_time_ids.repeat(bsz, 1)
                
                cond_kwargs = {
                    "text_embeds": pooled_embeds.to(dtype=weight_dtype), 
                    "time_ids": add_time_ids
                }

                # 5. U-Net Forward
                # Note: 'prompt_embeds' goes into cross-attention, 'cond_kwargs' goes into the add-on projection
                noise_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=cond_kwargs).sample

                # 6. Loss Calculation
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # 7. Zero out gradients for non-placeholder tokens
                if accelerator.num_processes > 1:
                    grads1 = text_encoder1.module.get_input_embeddings().weight.grad
                    grads2 = text_encoder2.module.get_input_embeddings().weight.grad
                else:
                    grads1 = text_encoder1.get_input_embeddings().weight.grad
                    grads2 = text_encoder2.get_input_embeddings().weight.grad
                
                # Zero out TE1
                index_grads_to_zero1 = torch.arange(len(XLtok1)) != placeholder_token_id
                grads1.data[index_grads_to_zero1, :] = grads1.data[index_grads_to_zero1, :].fill_(0)
                
                # Zero out TE2
                index_grads_to_zero2 = torch.arange(len(XLtok2)) != placeholder_token_id
                grads2.data[index_grads_to_zero2, :] = grads2.data[index_grads_to_zero2, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % hyperparameters["save_steps"] == 0:
                    save_progress(text_encoder1, text_encoder2, global_step, placeholder_token_id, accelerator, output_dir)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Save logic remains the same...
    if accelerator.is_main_process:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
                    pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(text_encoder1),
                    text_encoder_2=accelerator.unwrap_model(text_encoder2),
                    tokenizer=XLtok1,
                    tokenizer_2=XLtok2,
                    vae=vae,
                    unet=unet,
                    safety_checker=None
                    )
        pipeline.save_pretrained(output_dir)
        save_progress(text_encoder1, text_encoder2, global_step, placeholder_token_id, accelerator, output_dir)


import accelerate
accelerate.notebook_launcher(training_function, num_processes=1, args=(XLt1, XLt2, XLvae, XLunet))

for param in itertools.chain(XLunet.parameters(), XLt1.parameters(), XLt2.parameters()):
  if param.grad is not None:
    del param.grad  # free some memory
  torch.cuda.empty_cache()