# NSYNC with Textual Inversion training parameters
version = "v4.2"

# automatic flags for each settings!
if version == "v4.1":
    use_simclr = False  
    use_neg_prompts = False 
    usage_anchor = False  
elif version == "v4.2":
    use_simclr = False  
    use_neg_prompts = False
    usage_anchor = True 
elif version == "v5.0":
    use_simclr = False  
    use_neg_prompts = True
    usage_anchor = False 
elif version == "v5.1":
    use_simclr = False   
    use_neg_prompts = True  
    usage_anchor = True

sd_version = 'sd15' # or sd2.0

target_name = "monet2photo"  # ukiyoe2photo
split_set = 'trainA'
initializer_token = "illustration" # "illustration", "painting", "animation"  # TODO: adjust based on your dataset content!
generic_data_root = f"./datasets/generic_dataset_{sd_version}/{target_name}/{split_set}"
generic_placeholder_token = initializer_token

prompt_dir = f"./datasets/prompts/{target_name}/{split_set}"
images_path = f"./datasets/{target_name}/{split_set}"
save_path = f"./embeddings_{sd_version}/{version}/{target_name}"

what_to_teach = "style"
placeholder_token = "<illustration-special>"  # TODO: fixed no change!
if sd_version == 'sd15':
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
else:
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-base"

hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 8000,  # 2000
    "save_steps": 1000,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "output_dir": save_path
}

# generate NEG set: gen_generic_dataset.py
content_set = False  # for styleshot etc. normaly it is False for baseline_sd results and generic dataset (with a generic token)
generic_target_name = "monet2photo"
generic_split = "trainA"  # To create NEG set for training use trainA, to get the baseline SD results use testA
generic_prompts_dir = f"./datasets/prompts/{generic_target_name}/{generic_split}"
if content_set:
    generic_special_token = ""
    generic_split = "testA"
    generic_save_dir =  f"./datasets/generic_dataset_content_{sd_version}/{generic_target_name}/{generic_split}"
else:
    if generic_split == 'trainA':
        generic_special_token = initializer_token #  "Wassily Kandinsky" #"Claude Monet" # initializer_token  #"Studio Ghibli animation"  #"Cezanne" 
        generic_save_dir =  f"./datasets/generic_dataset_{sd_version}/{generic_target_name}/{generic_split}"
    else:
        generic_save_dir =  f"./results/baseline_{sd_version}/{generic_target_name}/{generic_split}"
        generic_special_token = "Claude Monet" #   #"Claude Monet", "Studio Ghibli animation", "Van Gogh" 
img_per_prompt = 1
img_size = 256

# Img2Text: img2txt_internvl.py
caption_target_name = "monet2photo"
prompts_input_dir = f"./datasets/{caption_target_name}"
prompts_out_dir = f"./datasets/prompts/{caption_target_name}"

# Inference: gen_imgs.py
img_per_prompt_inference = 1  # How many samples to generate for each test prompt.
result_type = "img2txt2img" # No need to change
test_target_name = "monet2photo"  # TODO: adjust based on your dataset
iter_num = 8000  # itreation to test
test_version = "v4.2"  # or v4.2 for NSYNC, v0.1 is for baseline TI
test_split = "testA"  # Split to use for inference.
test_model_file = f"learned_embeds-step-{iter_num}.bin"
test_embed_path = f"./embeddings_{sd_version}/{test_version}/{test_target_name}/{test_model_file}"  # bin model file
test_save_dir =  f"./results_{sd_version}/{test_version}/{result_type}/{test_target_name}_{iter_num}/"
test_prompts_dir = f"./datasets/prompts/{test_target_name}/{test_split}"  # testA folder

# Baseline TI: sd_textual_inversion_training.py
baseline_version = "v0.1" # not used in the training etc. Don't change
baseline_target_name = "monet2photo"

baseline_save_path = f"./embeddings_{sd_version}/{baseline_version}/{baseline_target_name}"
baseline_prompt_dir = f"./datasets/prompts/{baseline_target_name}/{split_set}"
baseline_images_path = f"./datasets/{baseline_target_name}/{split_set}"

hyperparameters_baseline_ti = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 8000,
    "save_steps": 1000,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "output_dir": baseline_save_path
}


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]
