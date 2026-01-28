# APSE-Net:

## Introduction

This is the official implementation of the paper, Few-Shot Segmentation of Mining Areas via Adaptive Prior Enhancement and Selective Edge Attention.

## Results and models
| Fold   | PT & LOG |
|--------|---------|
| fold 0 | [fold0_pt_log](https://pan.baidu.com/s/1ZYTvGZq8IHXd-gspsuCvCg?pwd=apse) |
| fold 1 | [fold1_pt_log](https://pan.baidu.com/s/17oh7ggJXmQemuvPeHJoLBw?pwd=apse) |
| fold 2 | [fold2_pt_log](https://pan.baidu.com/s/1SYstK6RS_FF0DRrGyGJPbw?pwd=apse) |

## Environment

```shell
python                   3.13.5
cuda                     12.1
torch                    2.6.0+cu118
torchaudio               2.6.0+cu118
torchvision              0.21.0+cu118
PyYAML                   6.0.2
albumentations           2.0.8
```

### DATA 
To Be open
In the Config file ./Configs/*.yaml change the data path ```data_root``` to ```YOUR_DATA_PATH```

## Usage

### Training
```
python main_train.py Config/FGE_ssblock_LCML_0.yaml
```
AMP:
```
python main_amp_train.py Config/APSE_fold_0.yaml
```

### Test

```
python main_test+.py Config/APSE_fold_0.yaml "PTH_PATH" "nSHOT(1 or 5)" 
example: python main_test+.py Config/APSE_fold_0.yaml logs/APSE_fold_0/APSE_fold_0_amp/best.pth 5 
```



