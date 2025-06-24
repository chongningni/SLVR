# SLVR: Super-Light Visual Reconstruction via Blueprint Controllable Convolutions and Exploring Feature Diversity Representation (CVPR 2025)
📖[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Ni_SLVR_Super-Light_Visual_Reconstruction_via_Blueprint_Controllable_Convolutions_and_Exploring_CVPR_2025_paper.html) |🖼️[PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_SLVR_Super-Light_Visual_Reconstruction_via_Blueprint_Controllable_Convolutions_and_Exploring_CVPR_2025_paper.pdf)

PyTorch codes for 《Super-Light Visual Reconstruction via Blueprint Controllable Convolutions and Exploring Feature Diversity Representation》, Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025.

Authors: Ning Ni and Libao Zhang (Beijing Normal University)


## Usage 🗝
```
git clone https://github.com/chongningni/SLVR.git
```
## Requirements 🛒
* pytorch==1.10.0
* pytorch-lightning==1.5.5
* matplotlib
* scikit_image
* numpy
* opencv-python
* easydict
* tqdm


## Pretrained checkpoints/models:
1. Model pretrained on the DIV2K/DF2K dataset: download from here [download model](https://github.com/chongningni/SLVR/releases/download/pretrain/pretrain_model.zip)
   



## Train👇
① Download training datset from here: [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

② Configure the yaml file according to your file path ( In config/ )

③ Start training with the following command:
```
python train.py --config config/config_file.yaml
```

## Test👇
① Download benchmark datasets Set5,Set14,B100,Urban100,Manga109 and put them on path: `load/benchmark/datset_name`
```
python test.py --checkpoint your_saving_checkpoint --datasets Set5,Set14,B100,Urban100,Manga109 --scales 4
```


## Citation🤝
If you find our work helpful in your research, please consider citing it. Thanks! 🤞
```
@InProceedings{Ni_2025_CVPR,
    author    = {Ni, Ning and Zhang, Libao},
    title     = {SLVR: Super-Light Visual Reconstruction via Blueprint Controllable Convolutions and Exploring Feature Diversity Representation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {400-410}
}
```


