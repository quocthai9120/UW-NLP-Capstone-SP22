# ClipCap++: Improvements to CLIP Prefix based Image Captioning

## This code is based on the official implementation for the paper ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734)

We thank the authors for their work and for sharing their ["implementation"]("https://github.com/rmokady/CLIP_prefix_caption")

## Setup
For evaluation we use the [COCO caption evaluation tool](https://github.com/LuoweiZhou/coco-caption/tree/de6f385503ac9a4305a1dcdc39c02312f9fa13fc), we suggest installing it via
```
pip install git+https://github.com/flauted/coco-caption.git@python23
```

For specific packages, we refer the user to our conda env file `environment.yml`

```
# TODO change to ours
git clone https://github.com/rmokady/CLIP_prefix_caption && cd UW-NLP-Capstone-SP22
conda env create -f environment.yml
conda activate clip_prefix_caption
```

## COCO training

Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/coco/annotations`.
Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip (We use Karpathy et el. split).

Extract CLIP features using (output is `data/coco/oscar_split_ViT-B_32_train.pkl`):
```
python parse_coco.py --clip_model_type ViT-B/32
```
Train with fine-tuning of GPT2:
```
python train.py --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/
```

Train only transformer mapping network:
```
python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40
```

For the annotation, we suggest downloading our copy of the [validation captions](https://drive.google.com/file/d/1AIE2eQlcyi46djvnDfMgI9IV5BYIEz4q/view?usp=sharing).

Our refinment requires first saving the predictions from base model, and saving the encoded text features from CLIP's text encoder:
```
CUDA_VISIBLE_DEVICES=0 python save_captions.py \
    --only_prefix \
    --mapping_type transformer \
    --num_layers 8 \
    --prefix_length 40 \
    --prefix_length_clip 40 \
    --run_type train \
    --tag clipcap_base \
    --clip_model_type ViT-B/32
```

This saves them as `data/coco/oscar_split_clipcap_base_train.pkl` (make sure to run above for train/val). To train the refinement text+image prefix model:
```
CUDA_VISIBLE_DEVICES=0 python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./refinement_v2-concat/ --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40 --bs 40 --prefix coco-prefix_refinment-v2-concat --text_data ./data/coco/oscar_split_clipcap_base_train.pkl
```

To evaluate the model we need to save predicions:
```
CUDA_VISIBLE_DEVICES=1 python predict.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_val.pkl --text_data ./data/coco/oscar_split_clipcap_base_val.pkl --out_dir ./refinement_v2-concat/ --mapping_type transformer --num_layers 8 --prefix_length 40 --prefix_length_clip 40 --weights ./refinement_v2-concat/coco-prefix_refinment-v2-concat_best.pt --tag best
```

Finally, run evaluation on the predictions by running:
```
python eval.py --preds_captions refinement_v2-concat/pred_val_caption_best.json
```

By design, the above process can be repeated to further refine model predictions e.g. text+image model can be repeated to get improved text features. However, this scales linearly to # iterations. While a valid future direction is to make the process more streamline i.e. removing recursive dependency of models, we note that is beyond the scope of this project.

**If you wish to use ResNet-based CLIP:** 
*Not implemented in our project.*
```
python parse_coco.py --clip_model_type RN50x4
```
```
python train.py --only_prefix --data ./data/coco/oscar_split_RN50x4_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40 --is_rn
```

## Citation
If you use our code for your research, please cite (along with original clipcap work):
```
# TODO: let's add our report here as well
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}
```
