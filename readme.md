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
1. Model pretrained on the DIV2K dataset: download from this [url]()
   
2. Model pretrained on the DF2K dataset: download from this [url]()


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
@articla{ni2024deformable,
    author={Ni, Ning and Zhang, Libao},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Deformable Convolution Alignment and Dynamic Scale-Aware Network for Continuous-Scale Satellite Video Super-Resolution}, 
    year={2024},
    volume={62},
    number={},
    pages={1-17},
    doi={10.1109/TGRS.2024.3366550}
}
```

```
N. Ni and L. Zhang, "Deformable Convolution Alignment and Dynamic Scale-Aware Network for Continuous-Scale Satellite Video Super-Resolution," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-17, 2024, Art no. 5610017, doi: 10.1109/TGRS.2024.3366550.
```

## Acknowledgement🙏
Our code is built upon [TDAN](https://github.com/YapengTian/TDAN-VSR-CVPR-2020).

## Update◐◓◑◒
Satellite data is continuously updated……………………………………………………………………………………