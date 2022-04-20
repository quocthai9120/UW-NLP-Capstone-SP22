from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')


def main() -> None:
    preds_captions = 'data/coco/annotations/pred_val_caption.json'
    true_captions = 'data/coco/annotations/captions_trainval2014.json'

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
