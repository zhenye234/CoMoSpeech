 

# COMOSPEECH

Implementation of the [CoMospeech](https://arxiv.org/pdf/2305.06908.pdf). For all details check out our paper accepted to ACM MM 2023: CoMoSpeech:  One-Step Speech and Singing Voice Synthesis via Consistency Model.

**Authors**: Zhen Ye, Wei Xue, Xu Tan, Jie Chen, Qifeng Liu, Yike Guo.

# Update

**2023-12-01**  

 

- We also propose a well-designed Singing Voice Conversion (SVC) version based on consistency model ([Code](https://github.com/Grace9994/CoMoSVC)). 

**2023-11-30**

- We find that zero-mean Gaussian noise instead of the prior in grad-tts can also achieve similar performance. We alse release the new code and checkpoints.

**2023-10-21** 

- We add Heun’s 2nd order method support for   teacher model (can be used for teacher model sampling and better ODE trajectory for consistency distillation). 

## Abstract

**Demo page**: [link](https://comospeech.github.io/).

Denoising diffusion probabilistic models (DDPMs) have shown promising performance for speech synthesis. However, a large number of iterative steps are required to achieve high sample quality, which restricts the inference speed. Maintaining sample quality while increasing sampling speed has become a challenging task. In this paper, we propose a **Co**nsistency **Mo**del-based Speech synthesis method, CoMoSpeech, which   achieve speech synthesis through a single diffusion sampling step while achieving high audio quality. The consistency constraint is applied to distill a consistency model from a well-designed diffusion-based teacher model, which ultimately yields superior performances in the distilled CoMoSpeech. 
Our experiments show that by generating audio recordings by a single sampling step, the CoMoSpeech achieves an inference speed more than 150 times faster than real-time on a single NVIDIA A100 GPU, which is comparable to FastSpeech2, making diffusion-sampling based speech synthesis truly practical. Meanwhile, objective and subjective evaluations on text-to-speech and singing voice synthesis show that the proposed teacher models yield the best audio quality, and the one-step sampling based CoMoSpeech achieves the best inference speed with better or comparable audio quality to other conventional multi-step diffusion model baselines.

## Prepare

Build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

 

## Inference

Run script `inference.py` by providing path to the text file, path to the  checkpoint, number of sampling :
```bash
    python inference.py -f <text file> -c <checkpoint> -t <sampling steps> 
```
Check out folder called `out` for generated audios. Note that in params file. Teacher = True is for our teacher model, False is for our ComoSpeech. In addition, we use the same vocoder in [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/). You can download it and put into checkpts folder.

 
## Training

We use LJSpeech datasets and follow the train/test/val split in fastspeech2, you can change the split in fs2_txt folder. Then run script `train.py` ,
```bash
    python train.py 
```
Note that in params file. Teacher = True is for our teacher model, False is for our ComoSpeech. While training Comospeech, teacher checkpoint directory should be provide.

Checkpoints trained on LJSpeech can be download from [here](https://drive.google.com/drive/folders/1rkbzl9NzS_fKtMubQ7FgSdgt7v8ZuYGk?usp=sharing).

## Acknowledgement
I would like to extend a special thanks to authors of Grad-TTS, since our code base is mainly borrowed from  [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/).

## Contact
You are welcome to send pull requests or share some ideas with me. Contact information: Zhen YE ( zhenye312@gmail.com )

