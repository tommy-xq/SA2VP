# SA2VP: Spatially Aligned-and-Adapted Visual Prompt

paper link: https://arxiv.org/abs/2312.10376

------

This repository contains the official PyTorch implementation for SA2VP.

![model_img](https://github.com/tommy-xq/SA2VP/blob/main/imgs/SA2VP.png)

## Environment settings

We use the framework from https://github.com/microsoft/unilm/tree/master/beit

we use following datasets for evaluation:

https://github.com/KMnP/vpt (FGVC)

https://github.com/dongzelian/SSF (VTAB-1k)

https://github.com/shikiw/DAM-VP (HTA)

This code is tested with Python-3.7.13, Pytorch = 1.12.1 and CUDA = 11.4, requiring the following dependencies:

* timm = 0.6.7

we also provide the requirement.txt for reference.

## Structure of this repo

- `./backbone_ckpt`: save the ViT and Swin Transformer pre-trained ckpt.

- `./data`: download and setup input datasets, containing fgvc and vtab-1k.
```
│SA2VP/
├──data/
│   ├──fgvc/
│   │   ├──CUB_200_2011/
│   │   ├──OxfordFlower/
│   │   ├──Stanford-cars/
│   │   ├──Stanford-dogs/
│   │   ├──nabirds/
│   ├──vtab-1k/
│   │   ├──caltech101/
│   │   ├──cifar/
│   │   ├──.......
├──backbone_ckpt/
│   ├──imagenet21k_ViT-B_16.npz
│   ├──swin_base_patch4_window7_224_22k.pth
```

- `./model_save`: save the final ckpt.

-  `./log_save`: save the log.


- `./vpt_main`: we use the [VPT](https://github.com/KMnP/vpt) code to initialize model.

    * 👉`./vpt_main/src/models/vit_backbones/vit_tinypara.py`: <u>SA2VP based on ViT backbone.</u> 

    * 👉`./vpt_main/src/models/vit_backbones/vit_tinypara_acc.py`: <u>We have accelerated the attention calculation of SA2VP.</u> 

    * 👉 `./vpt_main/src/models/vit_backbones/swin_transformer_tinypara.py`: <u>SA2VP based on Swin Transformer backbone.</u>

    * `./vpt_main/src/models/build_swin_backbone.py`: package SA2VP based on Swin. In this file, it will import model in swin_transformer_tinypara.py.

- `datasets.py`: contain all datasets.
- `engine_for_train.py`: engine for train and test.

- 👉`vit_train_sa2vp.py`: call this to train SA2VP based on ViT. In line 37, you can use the accelerated version by adding '_acc' to the model name.

- 👉`vit_train_swin.py`: call this to train SA2VP based on Swin Transformer.

- 👉`Train_nature.sh/Train_special.sh/Train_struct.sh`: scripts used for automatic training.

## Experiment steps

- 1\ Download the pre-trained ckpt of ViT and Swin from [VPT](https://github.com/KMnP/vpt). Use <u>ViT-B/16 Supervised</u> and <u>Swin-B Supervised</u>.

- 2\ Change the name and path in `vit_train_sa2vp.py` line 48 and in `vit_train_swin.py` line 47.

- 3\ Set different branch training weights in `engine_for_train.py` line 26/177.

- 4\ Set datasets path in `datasets.py` line 1160/1161 (prefix_fgvc/prefix_vtab). <u>Note that you need to choose transform for fgvc or vtab in line 1157/1158 and Pay attention to the dataset name in the following</u>.

- 5\ Change model config. For SA2VP based on ViT, we set <u>inter-dim</u> in `vit_tinypara.py` line 280/281/334/428 and <u>inter-weight</u> in line 427. For SA2VP based on Swin, set <u>inter-dim</u> in `vit_train_swin.py` line 169/170/675 and <u>inter-weight</u> in line 596. Default lr 1e-3 and weight_decay 1e-4.
- For ViT: (vtab: SVHN-16-0.5; Resisc45-16-0.5; ds/ori-16-0.1; sn/ele-32-0.5 need to Specially handle. || vtab special lr: Pets-5e-4; Clevr/Count-5e-4.)

| |CUB | Nabirds| Flower |  DOG| CAR |
|------| :--------: | :-----:  |  :-----:  | :-----:  | :-----:|
| inter-dim |16    | 32  | 8 |  32|64 | |
| inter-weight|0.1  | 0.1  | 0.1 |0.1 | 1.5| |
| batch size|64/128  | 64/128  | 64/128 |64/128 | 64/128| |

| |vtab-Natural | vtab-Special| vtab-Structure | HTA|
|------| :--------: | :-----:  |  :-----:  |:-----:  |
| inter-dim | 8   | 16 | 32 |  64| |
|inter-weight|0.1  | 1.5  | 1.5 | 0.1| |
|batch size|40/64  | 40  | 40 | 64/128| |

- For Swin:

| |vtab-Natural | vtab-Special| vtab-Structure |
|------| :--------: | :-----:  |  :-----:  |
| inter-dim | 8   | 8 | 8 |  
|inter-weight|0.1/0.5  | 0.5/1.5  | 1.5 | 
|batch size|40/64  | 40 | 40 | 

- Training Scripts:
  - Single GPU
  ```
  CUDA_VISIBLE_DEVICES=1 python vit_train_sa2vp.py  --data_set CUB --output_dir ./model_save/CUB --update_freq 1  --warmup_epochs 10 --epochs 100 --drop_path 0.0  --lr 1e-3 --weight_decay 1e-4 --nb_classes 200 --log_dir ./log_save --batch_size 64 --my_mode train_val --min_lr 1e-7
  ```
  - Multiple GPUs
  ```
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 vit_train_sa2vp.py  --data_set CIFAR --output_dir ./model_save/CIFAR --update_freq 1  --warmup_epochs 10 --epochs 100 --drop_path 0.0  --lr 1e-3 --weight_decay 1e-4 --nb_classes 100 --log_dir ./log_save --batch_size 40 --my_mode train_val --min_lr 1e-7
  ```

- Test Script:
  - For VTAB-1k
  ```
  CUDA_VISIBLE_DEVICES=1 python vit_train_sa2vp.py --data_set DS_LOC --eval --batch_size 64 --resume ./model_save/DS_LOC/checkpoint-99.pth --nb_classes 16 --my_mode trainval_test
  ```
  - For FGVC
  ```
  CUDA_VISIBLE_DEVICES=1 python vit_train_sa2vp.py --data_set CAR --eval --batch_size 64 --resume ./model_save/CAR/checkpoint-best.pth --nb_classes 196 --my_mode trainval_test
  ```

- Note: --my_mode is to decide train/val/test sets. In train_val: to find the best model on val set when training. In trainval_test: use train/val sets to train and report acc on test set. We follow the strategy of VPT.




## Citation

If you find our work helpful in your research, please cite it as:

```
will be shown soon.
```

## License
The code is released under MIT License (see LICENSE file for details).