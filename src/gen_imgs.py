import os
os.environ['HF_HOME'] = '/mnt/isilon/shicsonmez/cache/huggingface'
os.environ['TORCH_HOME'] = '/mnt/isilon/shicsonmez/cache/torch'
import torch
import PIL
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
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

prompts = os.listdir(prompts_dir)
os.makedirs(save_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    safety_checker=None

).to("cuda")

tt = torch.load(embed_path)
pipe.load_textual_inversion(embed_path)  # , token=cfg.placeholder_token

for prompt in tqdm(prompts):
    image_name = os.path.splitext(prompt)[0]
    #read prompt
    prompt_file = os.path.join(prompts_dir, prompt)
    with open(prompt_file, "r") as p_file:
        prompt = p_file.read()
    prompt += " in the style of " + special_token
    imgs = pipe(prompt, num_images_per_prompt=img_per_prompt, num_inference_steps=50, guidance_scale=7.5).images
    #save imgs
    for i,img in enumerate(imgs):
        if img_per_prompt > 1:
            save_path = os.path.join(save_dir, image_name + "_" + str(i) + ".png")
        else:
            save_path = os.path.join(save_dir, image_name + ".png")
        img = img.resize((img_size, img_size))
        img.save(save_path)
