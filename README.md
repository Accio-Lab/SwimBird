<div align="center">
<br>
<h3>SwimBird: Eciliting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs</h3>

[Jintao Tong](https://scholar.google.com/citations?user=T-0oVM4AAAAJ&hl=en)<sup>1</sup>, [Shilin Yan](https://scholar.google.com/citations?user=2VhjOykAAAAJ&hl=en)<sup>2†‡</sup>, [Hongwei Xue](https://scholar.google.com/citations?user=k5CJa5YAAAAJ&hl=zh-CN)<sup>2</sup>, [Xiaojun Tang]()<sup>2</sup>, [Kunyu Shi]()<sup>2</sup>, 
<br>
[Guannan Zhang]()<sup>2</sup>, [Ruixuan Li](https://scholar.google.com/citations?user=scAIu2MAAAAJ&hl=en&oi=ao)<sup>1‡</sup>, [Yixiong Zou](https://scholar.google.com/citations?user=VXxF0mcAAAAJ&hl=en&oi=ao)<sup>1‡</sup>

<div class="is-size-6 publication-authors">
  <p class="footnote">
    <span class="footnote-symbol"><sup>†</sup></span>Project Leader
    <span class="footnote-symbol"><sup>‡</sup></span>Corresponding author
  </p>
</div>

<sup>1</sup>Huazhong University of Science and Technology,  <sup>2</sup>Accio Team, Alibaba Group

<div align="center">

[![ArXiv](https://img.shields.io/badge/arXiv-2602.06040-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2602.06040)
[![Project](https://img.shields.io/badge/Project-SwimBird-pink?style=flat&logo=Google%20chrome&logoColor=pink')](https://accio-lab.github.io/SwimBird)
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Data-SwimBird_SFT_92K-orange)](https://huggingface.co/Accio-Lab/SwimBird-8B)
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-SwimBird_8B-orange)](https://huggingface.co/datasets/Accio-Lab/SwimBird-SFT-92K)
</div>
</div>




## 🔥 News

* **`2025.02.06`** 🚀 [Model](https://huggingface.co/Accio-Lab/SwimBird-8B) and [Dataset](https://huggingface.co/datasets/Accio-Lab/SwimBird-SFT-92K) are released!
* **`2025.02.05`** 🚀 [Training Code](https://github.com/Accio-Lab/SwimBird) is available!
* **`2025.02.05`** 📝 We release our latest work [SwimBird](https://arxiv.org/abs/2602.06040)!


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
2. Download the training datasets [SwimBird-SFT-92K]() and add the dataset absolute directory path as a prefix to all image paths in the JSON files:

```Shell
python data_process.py absolute_path_to_dataset
```

Example:
```Shell
python data_process.py /abs_path/SwimBird-ZebraCoT/
python data_process.py /abs_path/SwimBird-MathCanvas/
python data_process.py /abs_path/SwimBird-ThinkMorph/
python data_process.py /abs_path/SwimBird-OpenMMReasoner/
```


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


## ✉️ Concat
- If you have any questions about this project, please feel free to contact: tattoo.ysl@gmail.com.
- We are actively seeking self-motivated researchers and research interns to join our team!

## 📌 Citation
- If you find this project useful in your research, please consider citing:

```bibtex
arxiv
```


## 👍 Acknowledgment
- We sincerely thank [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune), [Skila](https://github.com/TungChintao/SkiLa) and others for their contributions, which have provided valuable insights.
