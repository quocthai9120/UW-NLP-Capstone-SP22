# Loss manager
# author: Brian Lee <joonl4@cs.washington.edu>
from torch.nn import functional as nnf

class LossManager():
    def __init__(self, weights=[1.0, 0.5]):
        self.weights = weights
        self.loss_outputs = ["caption_1_logits"]#, "caption_2_logits"]
    def get_loss(self, pred_dict, label):
        """
        Since the groundtruth remains the same, it is assumed only a single labeld is given.

        pred_dict: dictionary of model predictions
        label: caption label
        """
        loss_out = {}
        total_loss = 0.
        for w, k in zip(self.weights, self.loss_outputs):
            v = pred_dict[k]
            loss = nnf.cross_entropy(v.reshape(-1, v.shape[-1]), label.flatten(), ignore_index=0)
            l = k.replace("logits", "")
            loss_out[f"loss/{k}"] = loss
            total_loss += w * loss
        loss_out["loss/total"] = total_loss
        return loss_out