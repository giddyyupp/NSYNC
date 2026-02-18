import os
import torch
from tqdm import tqdm
from diffusers import FluxPipeline
import config as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompts_dir = cfg.flux_prompts_dir
img_size = cfg.img_size
special_token = cfg.flux_style_id

prompts = os.listdir(prompts_dir)
os.makedirs(cfg.flux_save_dir, exist_ok=True)

for ind, prompt in enumerate(tqdm(prompts)):
    image_name = os.path.splitext(prompt)[0]
    print(image_name)
    save_path = os.path.join(cfg.flux_save_dir, image_name + ".png")
    if os.path.exists(save_path):
        continue

    # read prompt
    prompt_file = os.path.join(prompts_dir, prompt)
    with open(prompt_file, "r") as p_file:
        prompt = p_file.read()
    
    prompt = "In the style of " + special_token + ", " + prompt
    print(prompt)
    img = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    img = img.resize((img_size, img_size))
    img.save(save_path)