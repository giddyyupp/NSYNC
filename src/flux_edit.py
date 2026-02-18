import os
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import config as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

img_size = cfg.img_size
os.makedirs(cfg.flux_save_dir, exist_ok=True)

test_images = os.listdir(cfg.flux_test_images)

for test_im in test_images:

    input_image = load_image(os.path.join(cfg.flux_test_images, test_im))

    img = pipe(
    image=input_image,
    prompt=f"Stylize image with {cfg.flux_style_id}",
    guidance_scale=2.5
    ).images[0]

    save_path = os.path.join(cfg.flux_save_dir, test_im + ".png")

    img = img.resize((img_size, img_size))
    img.save(save_path)