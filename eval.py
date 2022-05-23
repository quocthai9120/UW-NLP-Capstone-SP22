from predict import DATA_PATH
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder

DATA_PATH = '/local1/t3/data/coco/'
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def combine_mscoco_val_train_caption():
    train = json.load(open(DATA_PATH + 'annotations/captions_train2014.json', 'r'))
    val = json.load(open(DATA_PATH + 'annotations/captions_val2014.json', 'r'))
    train['annotations'].extend(val['annotations'])
    print("Combine json has", len(train['annotations']), "annotations")
    json.dump(train, open(DATA_PATH + 'annotations/captions_trainval2014.json', 'w'))


def main() -> None:
    combine_mscoco_val_train_caption()

    preds_captions = DATA_PATH + 'annotations/pred_val_caption.json'
    true_captions = DATA_PATH + 'annotations/captions_val2014.json'

    coco = COCO(true_captions)
    valids = coco.getImgIds()

    preds = json.load(open(preds_captions, 'r'))

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
    main()
