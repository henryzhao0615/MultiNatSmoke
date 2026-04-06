# [WACV 2026] AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset

---

## Overview
<img src="assets/AusSmoke_vis.jpg" alt="Vis the dataset" width="1000"/>

The official implement for our WACV 2026 paper **AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset**. It provides:

- A new wildfire smoke segmentation dataset, named **MultiNatSmoke**, including smoke images around the world. 
- Evaluation scripts for state-of-the-art segmentation models on MultiNatSmoke.
- Baseline results and benchmark metrics.

---

## Datasets

Our dataset is partially compiled from various existing public datasets, with **added segmentation labels**. Please ensure you **cite the original datasets** before downloading or using our dataset. Most datasets are included in this release; however, the **Forest Fire** dataset requires a separate download. Use the script `curate_kaggle_forest_fire.py` (included in both the `code` folder and dataset release) for downloading and extracting the images.  

| Dataset | Link | License |
|---------|------|---------|
| **FIgLib** | [Link](https://www.hpwren.ucsd.edu/FIgLib/) | – |
| **Smoke5K** | [Link](https://github.com/SiyuanYan1/Transmission-BVM) | – |
| **SmokeSeg** | [Link](https://github.com/LujianYao/FoSp) | – |
| **AI-for-Mankind** | [Link](https://github.com/aiformankind/wildfire-smoke-dataset?tab=readme-ov-file) | CC BY-NC-SA 4.0 |
| **Firecam** | [Link](https://github.com/open-climate-tech/firecam/tree/master/datasets/2019a) | CC BY-NC-SA 4.0 |
| **Boreal Forest Fire** | [Link](https://etsin.fairdata.fi/dataset/1dce1023-493a-4d63-a906-f2a44f831898) | CC BY 4.0 |
| **D-Fire** | [Link](https://github.com/gaia-solutions-on-demand/DFireDataset) | CC0 1.0 (Public Domain) |
| **WSDataset** | [Link](https://www.kaggle.com/datasets/gloryvu/wildfire-smoke-detection) | MIT |
| **FireSpot** | [Link](https://github.com/Biometrix-4/FireSpot-CNX) | CC BY 4.0 |
| **FESB-MLID** | [Link](http://wildfire.fesb.hr/index.php?option=com_content&view=article&id=66%20&Itemid=76) | – |
| **Forest Fire** | [Link](https://www.kaggle.com/datasets/kutaykutlu/forest-fire) | – |

> **Note:** Please use the references listed in the **Citation** section of this repository when citing these datasets.

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

   * Download AnySmoke Dataset at [[Hugging Face 🤗]](https://huggingface.co/datasets/hongjinzhao0615/MultiNatSmoke)

2. **Train a model**

   ```bash
   python model_name/train.py
   ```

  * Model performance will be evaluated used IoU, MSE, F1, Precision and Recall.

---


## Result

1. **Performance on state-of-the-art segmentation models**
  <img src="assets/sota_performance.jpg" alt="Performance on SOTA models" width="700"/>
  <img src="assets/sota_performance_size.jpg" alt="Performance on SOTA models with different size" width="1000"/>

2. **Performance across varying training data percentages**
  <img src="assets/vis.jpg" alt="Visulization" width="1000"/>

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

