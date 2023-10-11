from torch import Tensor
import torch
import torch.nn.functional as F

class OHEMImageSampler:
    def __init__(self, min_kept: int, thresh: float, ignore_index: int = 255):
        self.min_kept = min_kept
        self.thresh = thresh
        self.ignore_index = ignore_index

    def sample(self, logits: Tensor, labels: Tensor):
        with torch.no_grad():
            prob = F.softmax(logits, dim=1)
            # 过滤ignore_index
            valid1 = labels != self.ignore_index
            prob = prob[valid1]

            # 取正确类别的分数
            tmp_prob = prob.gather(1, labels[valid1].unsqueeze(1)).squeeze(1)
            sort_prob, sort_indices = tmp_prob.sort()

            min_thresh = sort_prob[min(self.min_kept, sort_prob.numel()-1)]
            threshold = max(min_thresh, self.thresh)

            temp_valid = sort_prob < threshold
            valid_indices = sort_indices[temp_valid]

            valid2 = torch.zeros_like(labels, dtype=torch.bool)
            valid2[valid_indices] = True

            return valid1 & valid2