# NSYNC: Negative Synthetic Image Generation for Contrastive Training to Improve Stylized Text-To-Image Translation


<!-- [Conference/Journal Name], [Year]   -->

[Serkan Ozturk<sup>1</sup>](https://www.linkedin.com/in/serkanozturk97),
[Samet Hicsonmez<sup>2</sup>](https://scholar.google.com/citations?user=biHfDhUAAAAJ&hl),
[Pinar Duygulu<sup>1</sup>](https://scholar.google.com/citations?user=1KEMrHkAAAAJ&hl)

[<sup>1</sup>Department of Computer Engineering, Hacettepe University](https://cs.hacettepe.edu.tr), 

[<sup>2</sup>Interdisciplinary Centre for Security, Reliability, and Trust (SnT), University of Luxembourg](https://www.uni.lu/snt-en/research-groups/cvi2/), 

<!-- [![arXiv](https://img.shields.io/badge/arXiv-PDF-red)](link_to_arxiv) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) -->

## 🧠 Abstract

Current text conditioned image generation methods
output realistic looking images, but they fail to capture specific
styles. Simply finetuning them on the target style datasets still
struggles to grasp the style features. In this work, we present a
novel contrastive learning framework to improve the stylization
capability of large text-to-image diffusion models. Motivated by
the astonishing advance in image generation models that makes
synthetic data an intrinsic part of model training in various
computer vision tasks, we exploit synthetic image generation in
our approach. Usually, the generated synthetic data is dependent
on the task, and most of the time it is used to enlarge the
available real training dataset. With NSYNC, alternatively, we
focus on generating negative synthetic sets to be used in a novel
contrastive training scheme along with real positive images. In
our proposed training setup, we forward negative data along
with positive data and obtain negative and positive gradients,
respectively. We then refine the positive gradient by subtracting
its projection onto the negative gradient to get the orthogonal
component, based on which the parameters are updated. This
orthogonal component eliminates the trivial attributes that are
present in both positive and negative data and directs the
model towards capturing a more unique style. Experiments
on various styles of painters and illustrators show that our
approach improves the performance over the baseline methods
both quantitatively and qualitatively. Our code is available at
https://github.com/giddyyupp/NSYNC.


## 🖥️ Quick Start

### 1. Clone and Install


```bash
git clone https://github.com/giddyyupp/NSYNC
cd nsync

conda create -n nsync python=3.10
conda activate nsync
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download and organize the dataset as follows (e.g., monet2photo, vangogh2photo):

```
|-- ./datasets
    |-- monet2photo
        |-- testA
            |-- 000_train.png
        |-- trainA
            |-- 100_test.png
    |-- vangogh2photo
        |-- testA
            |-- 001.png
        |-- trainA
            |-- 002.png
    |-- other_datasets

```

---

### 3. Training and Inference


#### 3.1. Extract Descriptions from Images

We use a central config [file](src/config.py) to adjust parameters for different stages of the dataset preperation and training. 
First, in order to extract the descriptions from images, set `caption_target_name` in config file and run following command:

```bash
python img2txt_internvl.py
```

#### 3.2. Generate Negative Set
 
Next, to generate a negative set to be used during training, adjust the parameters under `generate NEG set:` section in config file and run following command:

```bash
python gen_generic_dataset.py
```

#### 3.3. Train NSYNC
 
Now we are ready to start training of NSYNC. Update `target_name` to your dataset name and `initializer_token` to the content of your dataset, e.g., painting or illustration, and run following command:

```bash
python contrastive_training.py
```

Also you can experiment with SD versions by setting `sd_version` parameter.

#### 3.4. Inference
 
To generate images with the trained model, first adjust the parameters under `Inference: gen_imgs.py` section, and run following command:

```bash
python gen_imgs.py
```


#### 3.5. Train baseline Textual Inversion
 
You could also train the baseline Textual Inversion model for comparison. Update `baseline_target_name` to your dataset name and `initializer_token` to the content of your dataset, e.g., painting or illustration, and run following command:

```bash
python sd_textual_inversion_training.py
```

Also you can experiment with SD versions by setting `sd_version` parameter.


#### 3.6. Metric Calculation

**CSD:**

We share the required files to use with [CSD](https://github.com/learn2phoenix/CSD) repo in the [metrics](./src/metrics) folder.
After setting up the CSD repo, first update the parameters (path to real and generated images etc.) in the `calculate_csd.py` then run the following command:

```bash
python calculate_csd.py --dataset nsync --model_path ./pretrainedmodels/pytorch_model.bin --gpu 0
```

**CMMD:**

For CMMD, we use the Pytorch [implementation](https://github.com/sayakpaul/cmmd-pytorch).

**FID and KID:**

FID and KID metrics are calculated using this [repo](https://github.com/abdulfatir/gan-metrics-pytorch).


---


## 📜 Citation

If you find this work useful, please cite:

<!-- ```bibtex
@inproceedings{nsync,
  title={NSYNC: Negative Synthetic Image Generation for Contrastive Training to Improve Stylized Text-To-Image Translation},
  author={},
  booktitle={TODO},
  year={2025}
}
``` -->

---

## 🪪 License

This repository is licensed under the Apache License. See the [`LICENSE`](LICENSE) file for details.

---

## 🙏 Acknowledgements

This repo builds upon open-source contributions from:

* [Textual Inversion](https://github.com/rinongal/textual_inversion)
* [Diffusers](https://github.com/huggingface/diffusers)


