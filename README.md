<div align="center">

# Laughing Matters

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2305.08854-B31B1B.svg)](https://arxiv.org/abs/2305.08854)
</div>

## Description

Official PyTorch implementation of **["Laughing Matters: Introducing Laughing-Face Generation using Diffusion Models"](https://arxiv.org/abs/2305.08854)**.

[Antoni Bigata Casademunt](https://scholar.google.com/citations?user=LuIdiV8AAAAJ&hl=en&oi=ao),
[Rodrigo Mira](https://scholar.google.com/citations?user=08YfKjcAAAAJ&hl=es&oi=ao),
[Nikita Drobyshev](https://scholar.google.com/citations?user=itNst7wAAAAJ&hl=en),
[Konstantinos Vougioukas](https://scholar.google.co.uk/citations?user=WwLpK44AAAAJ&hl=en),
[Stavros Petridis](https://scholar.google.co.uk/citations?user=6v-UKEMAAAAJ&hl=en),
[Maja Pantic](https://scholar.google.co.uk/citations?user=ygpxbK8AAAAJ&hl=en)
Imperial College London

### Abstract
Speech-driven animation has gained significant traction in recent years, with current methods achieving near-photorealistic results. However, the field remains underexplored regarding non-verbal communication despite evidence demonstrating its importance in human interaction. In particular, generating laughter sequences presents a unique challenge due to the intricacy and nuances of this behaviour. This paper aims to bridge this gap by proposing a novel model capable of generating realistic laughter sequences, given a still portrait and an audio clip containing laughter. We highlight the failure cases of traditional facial animation methods and leverage recent advances in diffusion models to produce convincing laughter videos. We train our model on a diverse set of laughter datasets and introduce an evaluation metric specifically designed for laughter. When compared with previous speech-driven approaches, our model achieves state-of-the-art performance across all metrics, even when these are re-trained for laughter generation.

## How to run

### 1. Environment setup

```bash

conda  create  -n  laughter  python=3.9  -y
conda  activate  laughter
pip  install  -r  requirements.txt

```

### 2. Dataset structure

 The fomat of the dataset needs to follow this structure:
 ```
Dataset
|-- VideoFolder
    |-- video1.avi
    |-- video2.avi
    |-- ...
|-- AudioFolder
    |-- video1.wav
    |-- video2.wav
    |-- ...
    |-- ...
```
Then in ./configs/datamodule/video_datamodule_universal_audio.yaml you can change the name of the audio and video folder as well as the extension of the videos and the corresponding audio files.

The dataloader will then only need a filelist of all the videos. To create the filelist, you can use:
```bash
python ./scripts/create_filelist.py --root_dir [ROOT_DATASET] \
	--dest_file [DESTINATION_FILE] \
	--ext [EXTENSION_VIDEOS]
```

### 3. Training

Execute the following script:

```bash
python src/train.py model=video_diffusion_edm \
	datamodule=video_datamodule_universal_audio \
	datamodule.batch_size=2  \
	datamodule.file_list_test=filelist_test.txt \ 
	datamodule.file_list_train=filelist_train.txt \
	datamodule.file_list_val=filelist_val.txt \
```

Then the script will run the model with the default parameters and store the logs in the logs folder. 
Have a look at *configs/model/defaults* for additional parameters.
We use [Hydra](https://hydra.cc/), so all parameters can be changed/added through the command line.

### 4. Eval

Execute the following script:

```bash
python src/eval.py ckpt_path=[CHECKPOINT_FOLDER] \
	model.autoregressive_passes=[N_AUTOREGRESSIVE_PASSES] \
	model.diffusion_model.num_sample_steps=[INFERENCE_DIFFUSION_STEPS] \
	datamodule.num_frames=16 datamodule.step=16  model.num_frames_val=16
```
The *ckpt_path* can be either the exact checkpoint or a folder containing the checkpoints. In the latter case, the script will look for the more recent checkpoint.

### 5. Inference

It is also possible to run the model on your own image and audio. For that, run the following script:

```bash
python src/generate.py model.num_frames_val=16 \
	ckpt_path=[CHECKPOINT_FOLDER] \
	autoregressive_passes=[N_AUTOREGRESSIVE_PASSES] \
	audio_path=[AUDIO_PATH] \
	image_path=[IMAGE_PATH] \
	model.diffusion_model.num_sample_steps=[INFERENCE_DIFFUSION_STEPS] \
```

 

### Citation

```bibtex
@article{DBLP:journals/corr/abs-2305-08854,
		author = {Antoni Bigata Casademunt and
			  Rodrigo Mira and
		          Nikita Drobyshev and
			  Konstantinos Vougioukas and
			  Stavros Petridis and
			  Maja Pantic},
		title = {Laughing Matters: Introducing Laughing-Face Generation using Diffusion Models},
		journal = {CoRR},
		volume = {abs/2305.08854},
		year = {2023}
}
```

  

### Note

  

There is a chance that the results presented in the paper may not be replicated precisely by this code, as there could be inadvertent human errors in the process of preparing and cleaning the code for release. If you face any challenges while attempting to reproduce our findings, we kindly request that you notify us without hesitation.

  

### Reference

This code is mainly built upon [make-a-video-pytorch](https://github.com/lucidrains/make-a-video-pytorch) and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repositories.\

Big thanks to [lucidrains](https://github.com/lucidrains) for providing high-quality open-source code.
