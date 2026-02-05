<div align="center">
  <h1 style="display: inline-block; margin: 0;">SwimBird: Eciliting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs
  </h1>
</div>


<h4 align="center"> 
Jintao Tong<sup>1</sup>,
Shilin Yan<sup>2†‡</sup>, 
Hongwei Xue<sup>2</sup>, 
Xiaojun Tang<sup>2</sup>, 
Kunyu Shi<sup>2</sup>
<br>
Guannan Zhang<sup>2</sup>,
Ruixuan Li<sup>1‡</sup>,
Yixiong Zou<sup>1‡</sup>
<br><br>
<sup>1</sup>Huazhong University of Science and Technology,  <sup>2</sup>Accio, Alibaba Group
<br><br> 
<sup>†</sup>Project Leader,  <sup>‡</sup>Corresponding author

</h4>

<div align="center">

[![ArXiv](https://img.shields.io/badge/arXiv-ID-AD1C18.svg?logo=arXiv)](xxx)
[![Project](https://img.shields.io/badge/Project-SwimBird-pink?style=flat&logo=Google%20chrome&logoColor=pink')](https://accio-lab.github.io/SwimBird/)
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Data-SwimBird_SFT_92K-orange)](https://huggingface.co/Accio-Lab/SwimBird-8B)
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-SwimBird_8B-orange)](https://huggingface.co/datasets/Accio-Lab/SwimBird-SFT-92K)
</div>

## 🔥 News

* **`2025.02.06`** 🚀 [Model](https://huggingface.co/Accio-Lab/SwimBird-8B) and [Dataset](https://huggingface.co/datasets/Accio-Lab/SwimBird-SFT-92K) are released!
* **`2025.02.05`** 🚀 [Training Code](https://github.com/Accio-Lab/SwimBird) is available!
* **`2025.02.05`** 📝 We release our latest work [SwimBird](xxx)


## 🌟 Method
We introduce SwimBird, a hybrid autoregressive MLLM that dynamically switches among three reasoning modes conditioned on the input: (1) text-only reasoning, (2) vision-only reasoning (continuous hidden states
as visual thoughts), and (3) interleaved vision–text reasoning. By enabling flexible, query-adaptive mode selection, SwimBird preserves strong textual logic while substantially improving performance on vision-dense tasks.

<p align='center'>
<img src='https://github.com/Accio-Lab/SwimBird/blob/main/img/method.jpg' alt='mask' width='950px'>
</p>

## 👀 Cases
SwimBird dynamically switches among three reasoning modes conditioned on the input: (1) text-only reasoning, (2) vision-only reasoning, and (3) interleaved vision–text reasoning.

<p align='center'>
<img src='https://github.com/Accio-Lab/SwimBird/blob/main/img/case.jpg' alt='mask' width='950px'>
</p>


## 🛠 Preparation
```
git clone https://github.com/Accio-Lab/SwimBird.git
cd SwimBird

pip install -r requirements.txt
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

## 🎯 Training
To train the model, follow these steps:
1. Replace Qwen3-VL's `chat_template.json` with ours.
2. Download the training datasets [SwimBird-SFT-92K]()
3. Run the training script with the following command:

```Shell
bash scripts/train.sh
```

## 📖 Evaluation
We adopt [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to conduct the evaluation. You can get started as follows:
### 1. Setup 

```Shell
cd VLMEvalKit
pip install -e.
```

### 2. Inference

```Shell
bash test.sh
```

The path to our model: `VLMEvalKit-main/vlmeval/vlm/swimbird`

See [[QuickStar](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) | [快速开始](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/Quickstart.md)] for more details about arguments.


## 📌 Citation
- If you find this project useful in your research, please consider citing:

```bibtex
arxiv
```


## 👍 Acknowledgment
- We sincerely thank [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune), [Skila](https://github.com/TungChintao/SkiLa) and others for their contributions, which have provided valuable insights.
