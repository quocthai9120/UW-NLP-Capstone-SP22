_Disclaimer: This is a projected developed for a NLP Capstone Class at the University of Washington._

# ClipCap++: Efficient Image Captioning with CLIP

We propose an efficient image captioning model that utilizes pretrained Image and Language models. Our approach, based on a prior work(**ClipCap: CLIP Prefix for Image Captioning**), improves on the utilization of CLIP and GPT-2, showing competitive results on COCO Captions without fine-tuning any of the pretrained models.

## This code is based on the official implementation of ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734)

We thank the authors for their work and for sharing their [implementation](https://github.com/rmokady/CLIP_prefix_caption)

## Setup
For evaluation we use the [COCO caption evaluation tool](https://github.com/LuoweiZhou/coco-caption/tree/de6f385503ac9a4305a1dcdc39c02312f9fa13fc), we suggest installing it via
```
pip install git+https://github.com/flauted/coco-caption.git@python23
```

For specific packages, we refer the user to our conda env file `environment.yml`

```
git clone https://github.com/quocthai9120/UW-NLP-Capstone-SP22.git && cd UW-NLP-Capstone-SP22
conda env create -f environment.yml
conda activate clip_prefix_caption
```

## COCO training

Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/coco/annotations`.
Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip (We use Karpathy et el. split). Additionally, we suggest downloading our copy of the [validation captions](https://drive.google.com/file/d/1AIE2eQlcyi46djvnDfMgI9IV5BYIEz4q/view?usp=sharing).
Place the data into a directory named `data/` within the base directory of this repo.
### Extract CLIP features
output is `data/coco/oscar_split_<model_type>_<run_type>.pkl`, we support [`ViT-B_32`, `RN50x4`].
```
# for training data
python parse_coco.py --clip_model_type <model_type> --run_type train
# for validation data
python parse_coco.py --clip_model_type <model_type> --run_type val
```
### Training the mapping network
While the original ClipCap framework has two variants: MLP with finetuned GPT-2, mapping transformers with no finetuning of GPT-2, we focus on the latter. To train the transformer mapping network:
```
python train.py --only_prefix --data ./data/coco/oscar_split_<model_type>_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40
```

### Training the Spatial Feature Extraction model:
```
TODO
```

### Evaluation

To evaluate the model we need to save predicions:
```
CUDA_VISIBLE_DEVICES=1 python predict.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_val.pkl --text_data ./data/coco/oscar_split_clipcap_base_val.pkl --out_dir ./refinement_v2-concat/ --mapping_type transformer --num_layers 8 --prefix_length 40 --prefix_length_clip 40 --weights ./refinement_v2-concat/coco-prefix_refinment-v2-concat_best.pt --tag best
```

Finally, run evaluation on the predictions by running:
```
python eval.py --preds_captions refinement_v2-concat/pred_val_caption_best.json
```

### Guided Decoding
```
TODO
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
