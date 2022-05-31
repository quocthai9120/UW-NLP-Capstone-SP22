from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
import argparse

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def combine_mscoco_val_train_caption():
    train = json.load(open('data/coco/annotations/captions_train2014.json', 'r'))
    val = json.load(open('data/coco/annotations/captions_val2014.json', 'r'))
    train['annotations'].extend(val['annotations'])
    print("Combine json has", len(train['annotations']), "annotations")
    json.dumps('data/coco/annotations/captions_trainval2014.json')


def main(args) -> None:
    combine_mscoco_val_train_caption()

    # preds_captions = 'data/coco/annotations/pred_val_caption.json'
    # preds_captions = "data/coco/annotations/pred_val_caption_refinement-v1.json"
    # preds_captions = "data/coco/annotations/pred_val_caption_refinement-v1-concat.json" # epoch 6
    # preds_captions = "data/coco/annotations/pred_val_caption_refinement-v1-concat-best.json" # best epoch
    # preds_captions = "data/coco/annotations/pred_val_caption_baseline.json"
    preds_captions = args.preds_captions
    true_captions = 'data/coco/annotations/captions_val2014.json'

    coco = COCO(true_captions)
    valids = coco.getImgIds()

    preds = json.load(open(preds_captions, 'r'))
    gt = json.load(open(true_captions, 'r'))

    for pred in preds:
        pred['image_id'] = int(pred['image_id'])

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('Using %d/%d predictions ...' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open('tmp.json', 'w')) # serialize to temporary json file. Sigh, COCO API...

    resFile = 'tmp.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # serialize to file, to be read from Lua
    json.dump(out, open(preds_captions + '_out.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_captions', type=str)
    args = parser.parse_args()
    main(args)
