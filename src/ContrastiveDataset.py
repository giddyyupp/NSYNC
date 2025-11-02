import argparse
import itertools
import math
import os
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import config as cfg


class ContrastiveTIDataset(Dataset):
    def __init__(
        self,
        data_root,
        generic_data_root,
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
        self.generic_data_root = generic_data_root
        self.prompt_dir = prompt_dir
        self.tokenizer = tokenizer
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

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        self.usage_anchor = cfg.usage_anchor

    def __len__(self):
        return self._length
    
    def prep_sample(self, img_name, neg_sample=False):
        example = {}
        try:
            image = Image.open(img_name)
        except:
            image = Image.open(os.path.splitext(img_name)[0]+ '.png')

        # get image name
        img_name = os.path.basename(img_name)

        placeholder_string = self.placeholder_token
        prompt_name = os.path.splitext(img_name)[0] + ".txt"
        prompt_path = os.path.join(self.prompt_dir, prompt_name)
        with open(prompt_path, "r") as f:
            prompt = f.read()
        style_prompt = prompt + f" in the style of {placeholder_string}"
        if neg_sample and cfg.use_neg_prompts:
            style_prompt = prompt + f" not in the style of {placeholder_string}"

        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        example["input_ids"] = self.tokenizer(
            style_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

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

    def __getitem__(self, i):
        
        img_name = self.image_paths[i % self.num_images]
        example_pos = self.prep_sample(img_name=img_name)
        
        img_f_name = os.path.basename(img_name)
        generic_img_path = os.path.join(self.generic_data_root, img_f_name)
        example_generic = self.prep_sample(img_name=generic_img_path, neg_sample=True)

        if self.usage_anchor:
            img_name_anchor = self.image_paths[random.randint(0, self.num_images-1) % self.num_images]
            example_anchor = self.prep_sample(img_name=img_name_anchor)
        else:
            example_anchor = {}
        
        return example_pos, example_generic, example_anchor
