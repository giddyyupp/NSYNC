import argparse
import os


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y", "on"}


def parse_args():
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument("--version", default=os.getenv("NSYNC_VERSION", "v4.2"))
    p.add_argument("--sd_version", default=os.getenv("SD_VERSION", "sdxl"),
                   choices=["sdxl", "sd20", "sd15", "sd35"])

    p.add_argument("--target_name", default=os.getenv("TARGET_NAME", "monet2photo"))
    p.add_argument("--split_set", default=os.getenv("SPLIT_SET", "trainA"))

    # Optional: override style label text you use for flux prompts, etc.
    p.add_argument("--flux_style_id", default=os.getenv("FLUX_STYLE_ID", "Claude Monet"))

    # Optional: override main_dir if you run from different working dirs
    p.add_argument("--main_dir", default=os.getenv("MAIN_DIR", "../.."))

    p.add_argument("--initializer_token", default=os.getenv("INIT_TOKEN", "painting"),
                   choices=["illustration", "painting", "animation"])
    
    # generate NEG set: gen_generic_dataset.py
    p.add_argument("--content_set", type=str2bool,
               default=str2bool(os.getenv("CONTENT_SET", "False")))
    p.add_argument("--generic_target_name", default=os.getenv("GENERIC_TARGET_NAME", "monet2photo"))
    p.add_argument("--generic_split", default=os.getenv("GENERIC_SPLIT", "trainA"))
    
    # Inference: gen_imgs.py
    p.add_argument("--result_type", default=os.getenv("RESULT_TYPE", "img2txt2img"))
    p.add_argument("--test_target_name", default=os.getenv("TEST_TARGET_NAME", "monet2photo"))
    p.add_argument("--iter_num", type=int, default=int(os.getenv("ITER_NUM", "8000")))
    p.add_argument("--test_version", default=os.getenv("TEST_VERSION", "v4.2"))
    p.add_argument("--test_split", default=os.getenv("TEST_SPLIT", "testA"))
    p.add_argument("--use_kohya_sd", type=str2bool,
               default=str2bool(os.getenv("KOHYA_SD", "False")))
    p.add_argument("--use_random_prompts", type=str2bool,
            default=str2bool(os.getenv("TEST_RANDOM_PROMPTS", "False")))

    # baseline TI
    p.add_argument("--baseline_version", default=os.getenv("BASELINE_VERSION", "v0.1"))
    p.add_argument("--baseline_target_name", default=os.getenv("BASELINE_TARGET_NAME", "monet2photo"))


    args, _ = p.parse_known_args()
    return args


ARGS = parse_args()

# NSYNC with Textual Inversion training parameters

version = ARGS.version
sd_version = ARGS.sd_version
main_dir = ARGS.main_dir
target_name = ARGS.target_name
split_set = ARGS.split_set
flux_style_id = ARGS.flux_style_id
initializer_token = ARGS.initializer_token

# version = 'v4.2' # TI version
# sd_version = 'sdxl' #  sdxl, sd20, sd15 or sd35
# main_dir = "../.."
# target_name = "monet2photo"  # ukiyoe2photo
# target_name = "ghibli_dataset"  # ukiyoe2photo
# split_set = 'trainA'
# initializer_token = "painting" # "illustration", "painting", "animation"  # TODO: adjust based on your dataset content!

######################################################################
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

generic_data_root = f"{main_dir}/datasets/generic_dataset_{sd_version}/{target_name}/{split_set}"
generic_placeholder_token = initializer_token

prompt_dir = f"{main_dir}/datasets/prompts/{target_name}/{split_set}"
images_path = f"{main_dir}/datasets/{target_name}/{split_set}"
save_path = f"{main_dir}/embeddings_{sd_version}/{version}/{target_name}"

what_to_teach = "style" # TODO: fixed no change!
placeholder_token = "<illustration-special>"  # TODO: fixed no change!
if sd_version == 'sd15':
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
elif sd_version == 'sd35':
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium"
elif sd_version == 'sd20':
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-base"
elif sd_version == 'sdxl':
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
else:
    pretrained_model_name_or_path = None
    print("SELECT A PROPER SD version!!")

hyperparameters = {
    "learning_rate": 1e-06,
    "scale_lr": False,
    "max_train_steps": 8000,  # 2000
    "save_steps": 1000,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "output_dir": save_path
}

# FLUX part
######################################################################
flux_test_images = f"{main_dir}/datasets/{target_name}/testA"
# flux_style_id = "Vincent van Gogh"
flux_style_id = "Claude Monet"
# flux_style_id = "Studio Ghibli animation"
flux_save_dir = f"{main_dir}/results/flux/{target_name}/testA"
flux_prompts_dir = f"{main_dir}/datasets/prompts/{target_name}/testA"

######################################################################
# Generate NEG set: gen_generic_dataset.py
# content_set = False  # for styleshot etc. normaly it is False for baseline_sd results and generic dataset (with a generic token)
# generic_target_name = "monet2photo"
# generic_split = "trainA"  # To create NEG set for training use 'trainA', to get the baseline SD results use 'testA'
content_set = ARGS.content_set
generic_target_name = ARGS.generic_target_name
generic_split = ARGS.generic_split

generic_prompts_dir = f"{main_dir}/datasets/prompts/{generic_target_name}/{generic_split}"
if content_set:
    generic_special_token = ""
    generic_split = "testA"
    generic_save_dir =  f"{main_dir}/datasets/generic_dataset_content_{sd_version}/{generic_target_name}/{generic_split}"
else:
    if generic_split == 'trainA':
        generic_special_token = initializer_token #  "Wassily Kandinsky" #"Claude Monet" # initializer_token  #"Studio Ghibli animation"  #"Cezanne" 
        generic_save_dir =  f"{main_dir}/datasets/generic_dataset_{sd_version}/{generic_target_name}/{generic_split}"
    else:
        generic_save_dir =  f"{main_dir}/results/baseline_{sd_version}/{generic_target_name}/{generic_split}"
        generic_special_token = "Studio Ghibli animation"  # "Claude Monet", "Studio Ghibli animation", "Vincent Van Gogh painting" 
img_per_prompt = 1
img_size = 256

######################################################################
# Img2Text: img2txt_internvl.py
caption_target_name = "monet2photo"
prompts_input_dir = f"{main_dir}/datasets/{caption_target_name}"
prompts_out_dir = f"{main_dir}/datasets/prompts/{caption_target_name}"

######################################################################
# Inference: gen_imgs.py
img_per_prompt_inference = 1  # How many samples to generate for each test prompt.
# result_type = "img2txt2img" # No need to change
# test_target_name = "monet2photo"  # TODO: adjust based on your dataset
# iter_num = 8000  # itreation to test
# test_version = "v4.2"  # or v4.2 for NSYNC, v0.1 is for baseline TI
# test_split = "testA"  # Split to use for inference.
result_type = ARGS.result_type
test_target_name = ARGS.test_target_name
iter_num = ARGS.iter_num
test_version = ARGS.test_version
test_split = ARGS.test_split
use_kohya_sd = ARGS.use_kohya_sd
use_random_prompts = ARGS.use_random_prompts

random_prompt_dir = '../prompts'

test_model_file = f"learned_embeds-step-{iter_num}.bin"
test_embed_path = f"{main_dir}/embeddings_{sd_version}/{test_version}/{test_target_name}/{test_model_file}"  # bin model file
if not use_kohya_sd:
    test_save_dir =  f"{main_dir}/results_{sd_version}/{test_version}/{result_type}/{test_target_name}_{iter_num}/"
else:
    test_save_dir =  f"{main_dir}/results_{sd_version}_kohya/{test_version}/{result_type}/{test_target_name}_{iter_num}/"

test_prompts_dir = f"{main_dir}/datasets/prompts/{test_target_name}/{test_split}"  # testA folder

# SDXL Embeddings
if sd_version == 'sdxl':
    if not use_kohya_sd:
        test_model_file1_sdxl = f"learned_embeds1-step-{iter_num}.bin"
        test_model_file2_sdxl = f"learned_embeds2-step-{iter_num}.bin"
        test_embed_path1_sdxl = f"{main_dir}/embeddings_{sd_version}/{test_version}/{test_target_name}/{test_model_file1_sdxl}"  # bin model file
        test_embed_path2_sdxl = f"{main_dir}/embeddings_{sd_version}/{test_version}/{test_target_name}/{test_model_file2_sdxl}"  # bin model file
    else:
        test_embed_path_kohya = f"{main_dir}/embeddings_sd_scripts/{test_target_name}.safetensors"  # bin model file

if use_random_prompts:
    test_save_dir =  f"{main_dir}/results_{sd_version}/{test_version}/{result_type}/{test_target_name}_{iter_num}_random_prompts/"


####################################################################################
# Baseline TI: sd_textual_inversion_training.py  or sdxl_textual_inversion_training.py
baseline_version = ARGS.baseline_version # "v0.1" # not used in the training etc. Don't change
baseline_target_name = ARGS.baseline_target_name  # "monet2photo"

baseline_save_path = f"{main_dir}/embeddings_{sd_version}/{baseline_version}/{baseline_target_name}"
baseline_prompt_dir = f"{main_dir}/datasets/prompts/{baseline_target_name}/{split_set}"
baseline_images_path = f"{main_dir}/datasets/{baseline_target_name}/{split_set}"

hyperparameters_baseline_ti = {
    "learning_rate": 1e-06,
    "scale_lr": False,
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
