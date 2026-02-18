import argparse
import itertools
import math
import os
import random
import logging
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel, StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from ContrastiveDataset import ContrastiveTIDataset

import config as cfg


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


initializer_token = cfg.initializer_token 

generic_data_root = cfg.generic_data_root
generic_placeholder_token = cfg.generic_placeholder_token

prompt_dir = cfg.prompt_dir
images_path = cfg.images_path

what_to_teach = cfg.what_to_teach
placeholder_token = cfg.placeholder_token
pretrained_model_name_or_path = cfg.pretrained_model_name_or_path

imagenet_templates_small = cfg.imagenet_templates_small
imagenet_style_templates_small =cfg.imagenet_style_templates_small


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


train_dataset = ContrastiveTIDataset(
      data_root=cfg.images_path,
      generic_data_root = generic_data_root,
      prompt_dir = prompt_dir,
      tokenizer=XLtok1,
      size=XLvae.sample_size,
      placeholder_token=placeholder_token,
      repeats=100,
      learnable_property=what_to_teach, #Option selected above between object and style
      center_crop=False,
      set="train",
)


def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
# def create_orig_illustration_loader(train_batch_size=1):
#     return torch.utils.data.DataLoader(orig_illustration_dataset, batch_size=train_batch_size, shuffle=True)

noise_scheduler = XLsch #DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

hyperparameters = cfg.hyperparameters

def tensor_projection(A, B):
    """
    Calculate the projection of tensor A onto tensor B
    """
    # print("1st tensor shape: ", A.shape)
    # print("2nd tensor shape: ", B.shape)
    # Calculate dot product of A and B
    dot_product = torch.dot(A.view(-1), B.view(-1))
    
    # Calculate dot product of B with itself
    b_norm_sq = torch.dot(B.view(-1), B.view(-1))
    
    # Calculate projection
    projection = dot_product / b_norm_sq * B
    
    return projection


logger = get_logger(__name__)


# Helper to get the correct time_ids for SDXL
def compute_time_ids(original_size, crops_coords_top_left, target_size, device, dtype):
    # SDXL expects a tensor of shape (batch, 6) containing:
    # (original_height, original_width, crop_top, crop_left, target_height, target_width)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], device=device, dtype=dtype)
    return add_time_ids


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


def pred_noise(batch, vae, weight_dtype, text_encoder1, text_encoder2, default_time_ids, unet, accelerator):
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
    enc1_hidden = enc1_output.hidden_states[-2] 

    # Tokenizer 2 (OpenCLIP ViT-G)
    tokens2 = XLtok2(batch['prompt'], padding="max_length", max_length=XLtok2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
    enc2_output = text_encoder2(tokens2, output_hidden_states=True)
    # Use penultimate layer hidden states
    enc2_hidden = enc2_output.hidden_states[-2]
    # Get pooled output (for time_ids conditioning)
    pooled_embeds = enc2_output.text_embeds # Size: [batch, 1280]

    # Concatenate the two text encoders (Feature dimension: 768 + 1280 = 2048)
    prompt_embeds = torch.cat([enc1_hidden, enc2_hidden], dim=-1).to(dtype=weight_dtype)

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

    return noise, noise_pred, latents, timesteps


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
        for step, (batch, batch_neg_example, batch_anchor) in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder1), accelerator.accumulate(text_encoder2):
                
                ####### Positive Sample Part   ###########
                ##################################
                noise, noise_pred, latents, timesteps = pred_noise(batch, vae, weight_dtype, text_encoder1, text_encoder2, default_time_ids, unet, accelerator)
 
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
                grads1_temp = grads1.clone()
                grads2_temp = grads2.clone()


                ####### Negative Sample Part   ###########
                ##################################
                noise_orig_illust, noise_pred_orig_illust, latents_orig_illust, timesteps_orig_illust = pred_noise(batch_neg_example, vae, weight_dtype, text_encoder1, text_encoder2, default_time_ids, unet, accelerator)
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target_orig_illust = noise_orig_illust
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target_orig_illust = noise_scheduler.get_velocity(latents_orig_illust, noise_orig_illust, timesteps_orig_illust)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss_orig_illust = F.mse_loss(noise_pred_orig_illust, target_orig_illust, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss_orig_illust)

                # 7. Zero out gradients for non-placeholder tokens
                if accelerator.num_processes > 1:
                    grads1 = text_encoder1.module.get_input_embeddings().weight.grad
                    grads2 = text_encoder2.module.get_input_embeddings().weight.grad
                else:
                    grads1 = text_encoder1.get_input_embeddings().weight.grad
                    grads2 = text_encoder2.get_input_embeddings().weight.grad
                grads1_orig_illust_temp = grads1.clone()
                grads2_orig_illust_temp = grads2.clone()


                ####### Anchor Part   ###########
                ##################################

                if batch_anchor:
                    noise_anchor, noise_pred_anchor, latents_anchor, timesteps_anchor = pred_noise(batch_anchor, vae, weight_dtype, text_encoder1, text_encoder2, default_time_ids, unet, accelerator)
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target_anchor = noise_anchor
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target_anchor = noise_scheduler.get_velocity(latents_anchor, noise_anchor, timesteps_anchor)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss_anchor = F.mse_loss(noise_pred_anchor, target_anchor, reduction="none").mean([1, 2, 3]).mean()
                    accelerator.backward(loss_anchor)

                    # 7. Zero out gradients for non-placeholder tokens
                    if accelerator.num_processes > 1:
                        grads1 = text_encoder1.module.get_input_embeddings().weight.grad
                        grads2 = text_encoder2.module.get_input_embeddings().weight.grad
                    else:
                        grads1 = text_encoder1.get_input_embeddings().weight.grad
                        grads2 = text_encoder2.get_input_embeddings().weight.grad
                    grads1_anchor = grads1.clone()
                    grads2_anchor = grads2.clone()
                
                
                ########### orthogonality step
                ###########################################################################
                index_grads_placeholder_token = torch.arange(len(XLtok1)) == placeholder_token_id
                grads1_pos = grads1_temp[index_grads_placeholder_token]
                grads1_neg = grads1_orig_illust_temp[index_grads_placeholder_token]

                grads2_pos = grads2_temp[index_grads_placeholder_token]
                grads2_neg = grads2_orig_illust_temp[index_grads_placeholder_token]

                if cfg.usage_anchor:
                    grads1_pos_anchor = grads1_anchor[index_grads_placeholder_token]
                    grads2_pos_anchor = grads2_anchor[index_grads_placeholder_token]

                if not cfg.use_simclr: 
                    if cfg.use_neg_prompts:  # mean grad
                        if cfg.usage_anchor:
                            grad_update1 = (grads1_pos + grads1_pos_anchor + grads1_neg)/3.0  # TODO: try some weighting here!
                            grad_update2 = (grads2_pos + grads2_pos_anchor + grads2_neg)/3.0  # TODO: try some weighting here!
                        else:
                            grad_update1 = (grads1_pos + grads1_neg)/2.0  # TODO: try some weighting here!
                            grad_update2 = (grads2_pos + grads2_neg)/2.0  # TODO: try some weighting here!
                    else:  # grad proj
                        if cfg.usage_anchor:
                            grad1_projected_neg = tensor_projection(grads1_pos, grads1_neg)
                            grad1_projected_pos = tensor_projection(grads1_pos, grads1_pos)
                            grad_update1 = grads1_temp.data[index_grads_placeholder_token] + grad1_projected_pos - grad1_projected_neg 

                            grad2_projected_neg = tensor_projection(grads2_pos, grads2_neg)
                            grad2_projected_pos = tensor_projection(grads2_pos, grads2_pos)
                            grad_update2 = grads2_temp.data[index_grads_placeholder_token] + grad2_projected_pos - grad2_projected_neg 
                        else:
                            grad1_projected = tensor_projection(grads1_pos, grads1_neg)
                            grad_update1 = grads1_temp.data[index_grads_placeholder_token] - grad1_projected 

                            grad2_projected = tensor_projection(grads2_pos, grads2_neg)
                            grad_update2 = grads2_temp.data[index_grads_placeholder_token] - grad2_projected 

                    grads1.data[index_grads_placeholder_token] = grad_update1
                    grads2.data[index_grads_placeholder_token] = grad_update2
                else:  # not working ATM!
                    cont_loss_val = cont_loss(grads1_pos, grads1_pos_anchor, grads1_neg)  # gotta debug this one
                    accelerator.backward(cont_loss_val)
                    # grad_update = grads_temp.data[index_grads_placeholder_token] 

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
