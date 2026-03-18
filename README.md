# [WACV 2026] AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset

---

## Overview
<img src="assets/AusSmoke_vis.pdf" alt="Country-wise distribution of the AnySmoke dataset" width="1000"/>

The official implement for our WACV 2026 paper _“AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset”_. It provides:

- A new wildfire smoke segmentation dataset, named **MultiNatSmoke**, including smoke images around the world. 
- Evaluation scripts for state-of-the-art segmentation models on this dataset.
- Baseline results and benchmark metrics.

---

## Supported Models

We evaluate the following state-of-the-art segmentation models:

- **U-Net** (CNN-based)
- **DeepLabV3+** (CNN-based)
- **SegFormer** (transformer-based)
- **Mask2Former** (transformer-based)
- **FoSp** (domain-spefic)
- **Trans-BVM** (domain-spefic)

> The FoSp and Trans-BVM implementations follow their respective official repositories
   * **FoSp**: follow instructions and code at [LujianYao/FoSp](https://github.com/LujianYao/FoSp)
   * **Trans-BVM**: follow instructions and code at [SiyuanYan1/Transmission-BVM](https://github.com/SiyuanYan1/Transmission-BVM)

---

## Usage

1. **Prepare the AnySmoke dataset**

   * Download AnySmoke Dataset at [[Hugging Face 🤗]](https://www.kaggle.com/datasets/zhaohongjin0615/anysmoke)

2. **Train a model**

   ```bash
   python model_name/train.py
   ```

  * Model performance will be evaluated used IoU, MSE, F1, Precision and Recall.

---


## Result

1. **Performance on state-of-the-art segmentation models**
  <img src="assets/res_on_sota.jpg" alt="Country-wise distribution of the AnySmoke dataset" width="1000"/>

2. **Performance across varying training data percentages**
  <img src="assets/vis.jpg" alt="Country-wise distribution of the AnySmoke dataset" width="1000"/>

---

## Citation
If you find our work is useful for your research and works, please cite using this BibTeX:
```bibtex

@InProceedings{Li_2026_WACV,
    author    = {Li, Weihao and Zhao, Hongjin and Zhu, Gao and Ji, Ge-Peng and Wilson, Nicholas and Yebra, Marta and Barnes, Nick},
    title     = {AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {7996-8006}
}
```

