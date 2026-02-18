import os
import torch
import PIL
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
from tqdm import tqdm
import config as cfg

result_type = cfg.result_type
version = cfg.version

embed_path = cfg.test_embed_path
save_dir =  cfg.test_save_dir
prompts_dir = cfg.test_prompts_dir
special_token = cfg.placeholder_token # cfg.special_token -> for config2
pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
img_per_prompt = cfg.img_per_prompt_inference
img_size = cfg.img_size
use_kohya_sd = cfg.use_kohya_sd

test_target_name = cfg.test_target_name
prompt_type = 'paintings'

if test_target_name in ['monet2photo', 'vangogh2photo', 'pmondrian']:
    prompt_type = 'paintings'
elif test_target_name in ['ghibli_dataset']:
    prompt_type = 'animes'
else:
    prompt_type = 'illustrations'

prompts_f = os.path.join(cfg.random_prompt_dir, f"{prompt_type}.txt") 

with open(prompts_f, 'r') as ff:
    prompts = ff.readlines()

prompts = [prompt.replace('\n', '') for prompt in prompts]

os.makedirs(save_dir, exist_ok=True)

if cfg.sd_version == 'sdxl':
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype=torch.bfloat16,
        safety_checker=None
    ).to("cuda")
    # pipe.upcast_vae() # <--- Add this for better image quality
    # Load into the respective text encoders
    if not use_kohya_sd:
        pipe.load_textual_inversion(cfg.test_embed_path1_sdxl, token=cfg.placeholder_token, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
        pipe.load_textual_inversion(cfg.test_embed_path2_sdxl, token=cfg.placeholder_token, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
    else: # it is safetensors file
        from safetensors.torch import load_file
        state_dict = load_file(cfg.test_embed_path_kohya)
        pipe.load_textual_inversion(state_dict['clip_l'], token=cfg.placeholder_token, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
        pipe.load_textual_inversion(state_dict['clip_g'], token=cfg.placeholder_token, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    if not version == 'baseline':
        tt = torch.load(embed_path)
        pipe.load_textual_inversion(embed_path)

# 2. inference loop:
for ind, prompt in tqdm(enumerate(prompts)):
    image_name = f"img_{ind}"
    save_path = os.path.join(save_dir, image_name + ".png")
    if os.path.exists(save_path):
        continue

    # read prompt
    if not version == 'baseline':
        prompt = "In the style of " + special_token + ", " + prompt
    else:
        prompt = "In the style of " + cfg.flux_style_id + ", " + prompt

    print(prompt)

    with torch.no_grad():
        imgs = pipe(prompt, num_images_per_prompt=img_per_prompt, num_inference_steps=50, guidance_scale=7.5).images
    # save imgs
    for i,img in enumerate(imgs):
        if img_per_prompt > 1:
            save_path = os.path.join(save_dir, image_name + "_" + str(i) + ".png")
        else:
            save_path = os.path.join(save_dir, image_name + ".png")
        img = img.resize((img_size, img_size))
        img.save(save_path)
