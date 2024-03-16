# Zero-shot Improvement of Object Counting with CLIP
[Ruisu Zhang*](https://ruisu516.github.io/), [Yicong Chen*](https://bryce-chen.github.io/), [Kangwook Lee](https://kangwooklee.com/)

This is the code repository for paper **Zero-shot Improvement of Object Counting with CLIP** ([link](https://openreview.net/pdf?id=AJiBZ1BPH5)) accepted by R0-FoMo Workshop in Neurips 2023.


## Abstract
We focus on the object counting limitations of vision-language models, with a particular emphasis on Contrastive Language-Image Pre-Training (CLIP) models. 
We assess the counting performance of CLIP using a custom dataset, which uncovers significant variations across diverse objects. 
To address this, we introduce a zero-shot, training-free method aimed at improving counting accuracy by manipulating the text embedding space of CLIP. 
Through comprehensive experiments, we demonstrate that our method not only enhances the counting capabilities of CLIP but also boosts the performance of text-to-image generative models like Stable Diffusion, particularly in generating images with precise object counts.


## Reproduce experiment results
### Install required packaged
Create your own virtual environment using a python environment > 3.6
```
conda create -y -n ENV_NAME
conda activate ENC_NAME
cd $CODE_DIR
pip install -r requirements.txt
``` 
### Download dataset
Download our dataset [here](https://openreview.net/pdf?id=AJiBZ1BPH5).

### CLIP experiments
Run `python run.py` with specified configurations:
*  `--model`: choose CLIP model from ["clip_base_32","clip_base_16","clip_large_14"];
*  `--dataset`: choose dataset from ["custom","countbench"];
* `data_path`: specify the path to custom data if you run on the `custom` dataset;
* `eval_dir`: directory to save evaluation results
* `--task`: choose task from ["classification","image_retrievel"];
* `--ref_obj`: specify the name of reference object in a string format;

### Stable Diffusion experiments
Run `python run.py` with specified configurations:
* `-m`, `--model`: set as "stable_diffusion";
* `-t`, `--task`: set as "image_gen";
* `-r`, `--ref_obj`: specify the name of reference object in a string format;
