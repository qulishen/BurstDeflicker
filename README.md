## BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes ✨

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://arxiv.org/abs/2510.09996)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-green)](https://www.kaggle.com/datasets/lishenqu/burstflicker)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=qulishen.BurstDeflicker&right_color=violet)](https://github.com/qulishen/BurstDeflicker)
<!-- [![GitHub Stars](https://img.shields.io/github/stars/qulishen/BurstDeflicker?style=social)](https://github.com/qulishen/BurstDeflicker) -->

<p align="center">
  <img src="logo.png" width="1000px">
</p>

This repository contains the official implementation of our NeurIPS 2025 paper 📄:
<p>
<div><strong>BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes</strong></div>
<div>
  <a href="https://qulishen.github.io/">Lishen Qu</a>,
  <a href="https://qulishen.github.io/">Zhihao Liu</a>,
  <a href="https://joshyzhou.github.io/">Shihao Zhou</a>,
  <a href="https://qulishen.github.io/">Yaqi Luo</a>,
  <a href="https://liangjie.xyz/">Jie Liang</a>,
  <a href="https://huizeng.github.io/">Hui Zeng</a>,
  <a href="https://www4.comp.polyu.edu.hk/~cslzhang/">Lei Zhang</a>,
  <a href="https://cv.nankai.edu.cn/">Jufeng Yang</a>
</div>
<div>Accepted to <strong>NeurIPS 2025</strong> 🎉</div>

---

## TL;DR 🚀 | 快速开始

- **Train**: `bash ./dist_train.sh 2 options/Restormer.yml`
- **Inference**: `python test.py --input ... --output ... --model_path checkpoint/restormer.pth`
- **Evaluate**: `python evaluate.py --input ... --gt ...`

---

## Table of Contents 🧭 | 目录

- [Installation](#installation--安装)
- [Data Preparation](#data-preparation--数据准备)
- [Training](#training--训练)
- [Testing & Evaluation](#testing--evaluation--测试与评估)
- [Citation](#citation--引用)

---

## Installation 🛠️ | 安装

### 1) (Recommended) Create env 🐍

```bash
conda create -n burstdeflicker python=3.9 -y
conda activate burstdeflicker
```

### 2) Install dependencies 📦

```bash
pip install -r requirements.txt
```

### 3) Install BasicSR (develop mode) 🔧

Please run in the **repo root**:

```bash
python setup.py develop
```
---

## Data Preparation 🗂️ | 数据准备

Download dataset from Kaggle:
- 👉 [BurstFlicker Dataset](https://www.kaggle.com/datasets/lishenqu/burstflicker)

### Expected folder structure ✅

Training configs (e.g. `options/Restormer.yml`, `options/Burstormer.yml`) expect:

```text
dataset/
├── BurstFlicker-S/
│   ├── train-resize/
│   │   ├── input/
│   │   │   ├── 0001/ 0001.png 0002.png ... 
│   │   │   ├── 0002/ ...
│   │   │   └── ...
│   │   └── gt/
│   │       ├── 0001/ 0001.png 0002.png ...
│   │       └── ...
│   └── test-resize/
│       ├── input/ (same structure as train-resize)
│       └── gt/
└── BurstFlicker-G/
    ├── train-resize/
    │   ├── input/ (sequence folders)
    │   └── gt/
    └── test-resize/
        ├── input/
        └── gt/
```

---

## Training 🏋️ | 训练

### Restormer (default example) ✨

```bash
bash ./dist_train.sh 2 options/Restormer.yml
```
Other available configs:
- `options/Burstormer.yml`
- `options/HDRtransformer.yml`

```bash
 bash ./dist_train.sh 2 options/Restormer.yml
```

---

## Testing & Evaluation 🔎 | 测试与评估

### 1) Inference (generate restored frames) 🧪

`test.py` expects `--input` to be a folder that contains **multiple sequence subfolders**, e.g. `.../input/0001`, `.../input/0002`, ...

```bash
python test.py \
  --input dataset/BurstFlicker-S/test-resize/input \
  --output results/restormer \
  --model_path checkpoint/restormer.pth
```

### 2) Evaluate (PSNR / SSIM / LPIPS) 📊

```bash
python evaluate.py \
  --input results/restormer \
  --gt dataset/BurstFlicker-S/test-resize/gt
```

---

## Citation 📚 | 引用

If you find this work useful, please cite:

```bibtex
@inproceedings{BurstDeflicker_lishenqu,
    title={BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes},
    author={Lishen, Qu and Zhihao, Liu and Shihao, Zhou and Yaqi, Luo and Hui, Zeng and Lei, Zhang and Jie, Liang and Jufeng, Yang},
    booktitle={Advances in Neural Information Processing Systems},
    year={2025}
}
```

---
