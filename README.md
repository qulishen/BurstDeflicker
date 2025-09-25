# BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://arxiv.org/abs/xxxx.xxxxx)  
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-green)](https://www.kaggle.com/datasets/lishenqu/flarex)

This repository contains the official implementation of our NeurIPS 2025 paper:  
<p>
<div><strong>BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes</strong></div>
<div><a href="https://qulishen.github.io/">Lishen Qu</a>, 
   	<a href="https://qulishen.github.io/">Zhihao Liu</a>,
    <a href="https://joshyzhou.github.io/">Shihao Zhou</a>,
    <a href="https://qulishen.github.io/">Yaqi Luo</a>, 
    <a href="https://liangjie.xyz/">Jie Liang</a>
    <a href="https://huizeng.github.io/">Hui Zeng</a>
    <a href="https://www4.comp.polyu.edu.hk/~cslzhang/">Lei Zhang</a>
    <a href="https://cv.nankai.edu.cn/">Jufeng Yang</a>
    </div>
<div>Accepted to <strong>NeurIPS 2025</strong></div>

### Installation

1. Install dependent packages

    ```bash
    cd BurstFlicker
    pip install -r requirements.txt
    ```

2. Install Basicsr<br>
    Please run the following commands in the **root path** to install Basicsr:<br>

    ```bash
    python setup.py develop
    ```
### BurstFlicker dataset structure


[dataset link](https://www.kaggle.com/datasets/lishenqu/burstflicker)

```bash
├── dataset
    ├── BurstFlicker-G
        ├──train
            ├── input
            ├── gt
        ├──test
    ├── BurstFlicker-S
        ├──train
            ├── input
                ├──0001
                    ├──0001.png
                    ├──0002.png
                    ...
                    ├──0010.png
                ├──0002
                ...
            ├── gt
                ├──0001
                ├──0002
                ...
        ├──test
```

### Train on the synthetic dataset
```bash
bash ./dist_train.sh 2 options/Restormer_sys.yml
```

### Fine-tune on the real data 
```bash
bash ./dist_train.sh 2 options/Restormer.yml
```

### Test and evaluate
```bash
python test.py --input dataset/test/input --output result/restormer --model_path checkpoint/Restormer.pth
```

```bash
python evaluate.py --input result/restormer --gt dataset/test/gt
```
