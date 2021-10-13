# cross_view_localization_L2LTR

## will be updated soon

### Experiment Dataset

* CVUSA：[https://github.com/viibridges/crossnet](https://github.com/viibridges/crossnet)
* CVACT：[https://github.com/Liumouliu/OriCNN](https://github.com/Liumouliu/OriCNN)

### Dataset Preparation

Our method follow [SAFA](https://github.com/shiyujiao/cross_view_localization_SAFA) and [DSM](https://github.com/shiyujiao/cross_view_localization_DSM) to get polar transform aerial images. Please download ```data_prearation.py``` for pre-processing. 

### Models

Pretrained model Download(Google's Official Checkpoint)
* R50+ViT-B_16 [pretrained model](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)

Our trained models for CVUSA and CVACT will be available soon
* [Trained models](https://drive.google.com/file/d/1IOiElf_8-9Dq7n8vTAOi3kq8QAriFAjp/view?usp=sharing)


### Train and test model
Train
```
python train.py --name CVUSA --dataset CVUSA --pretrained_dir YOUR_PRETRAINED_MODEL --output_dir MODEL_WHERE_WILL_BE_SAVE --dataset_dir YOUR_DATASET_PATH --learning_rate 1e-4 --weight_decay 0.03
```
Test
```
python test.py --name CVUSA --dataset CVUSA --output_dir MODEL_WHERE_WILL_BE_SAVE --dataset_dir YOUR_DATASET_PATH
```
If you want to use auto mixed precision, you should install [APEX](https://github.com/NVIDIA/apex) and add ```--fp16``` in startup code.

### Result

|dataset|top-1|top-5|top-10|top-1%|
|:---|:---|:---|:---|:---|
|CVUSA|94.05%|98.27%|98.99%|99.67%|
|CVACT_val|84.89%|94.59%|95.96%|98.37%|
