import os
import numpy as np
from models.faceX.face_model import FeatureExtractor
from dataset.basedataset import PredictImageDatasets
from dataset.transforms import create_AugTransforms
from torch.utils.data import DataLoader

def process_pairtxt(pair_txt: str, imgdir: str):
    assert os.path.isfile(pair_txt), f'please check the path of {pair_txt}'

    pair_array= np.loadtxt(pair_txt, dtype=str)
    unique_face_images = np.unique(pair_array[:,:2].flatten()).tolist()
    unique_face_realpath = [os.path.join(imgdir,'val', path) for path in unique_face_images]

    pair_list = pair_array.tolist()

    return unique_face_realpath, pair_list
class Evaluator:

    def __init__(self, feature_extractor):
        """
        Args:
            feature_model: a feature extractor.
        """
        self.feature_extractor = feature_extractor

    def test(self, pair_list, feature_dataloader, device):
        # check pair_list
        Evaluator.check_nps(pair_list)
        image_name2feature = self.feature_extractor.extract_face(feature_dataloader, device)
        mean, std = self.test_one_model(pair_list, image_name2feature)
        return mean, std

    def test_one_model(self, test_pair_list, image_name2feature, is_normalize=True):
        """Get the accuracy of a model.

        Args:
            test_pair_list: the pair list given by PairsParser.
            image_name2feature: the map of image name and it's feature.
            is_normalize: wether the feature is normalized.

        Returns:
            mean: estimated mean accuracy.
            std: standard error of the mean.
        """
        nps = len(test_pair_list)
        nps_one_group = nps // 10
        subsets_score_list = np.zeros((10, nps_one_group), dtype=np.float32)
        subsets_label_list = np.zeros((10, nps_one_group), dtype=np.int8)
        for index, cur_pair in enumerate(test_pair_list):
            cur_subset = index // 600
            cur_id = index % 600
            image_name1 = os.path.normpath(cur_pair[0])
            image_name2 = os.path.normpath(cur_pair[1])
            label = cur_pair[2] if type(cur_pair[2]) is int else int(cur_pair[2])
            subsets_label_list[cur_subset][cur_id] = label
            feat1 = image_name2feature[image_name1]
            feat2 = image_name2feature[image_name2]
            if not is_normalize:
                feat1 = feat1 / np.linalg.norm(feat1)
                feat2 = feat2 / np.linalg.norm(feat2)
            cur_score = np.dot(feat1, feat2)
            subsets_score_list[cur_subset][cur_id] = cur_score

        subset_train = np.array([True] * 10)
        accu_list = []
        for subset_idx in range(10):
            test_score_list = subsets_score_list[subset_idx]
            test_label_list = subsets_label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = subsets_score_list[subset_train].flatten()
            train_label_list = subsets_label_list[subset_train].flatten()
            subset_train[subset_idx] = True
            best_thres = self.getThreshold(train_score_list, train_label_list)
            positive_score_list = test_score_list[test_label_list == 1]
            negtive_score_list = test_score_list[test_label_list == 0]
            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negtive_score_list < best_thres)
            accu_list.append((true_pos_pairs + true_neg_pairs) / 600)
        mean = np.mean(accu_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(10)  # ddof=1, division 9.
        return mean, std

    def getThreshold(self, score_list, label_list, num_thresholds=1000):
        """Get the best threshold by train_score_list and train_label_list.
        Args:
            score_list(ndarray): the score list of all pairs.
            label_list(ndarray): the label list of all pairs.
            num_thresholds(int): the number of threshold that used to compute roc.
        Returns:
            best_thres(float): the best threshold that computed by train set.
        """
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]
        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size
        score_max = np.max(score_list)
        score_min = np.min(score_list)
        score_span = score_max - score_min
        step = score_span / num_thresholds
        threshold_list = score_min + step * np.array(range(1, num_thresholds + 1))
        fpr_list = []
        tpr_list = []
        for threshold in threshold_list:
            fpr = np.sum(neg_score_list > threshold) / neg_pair_nums  # FP / [(FP + TN): all negative]
            tpr = np.sum(pos_score_list > threshold) / pos_pair_nums  # TP / [(TP + FN): all positive]
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        best_index = np.argmax(tpr - fpr)  # top-left in ROC-Curve
        best_thres = threshold_list[best_index]
        return best_thres

    @staticmethod
    def check_nps(pair_list):
        """check the number of pairs is a multiple of 10"""
        assert len(pair_list) % 10 == 0, 'make sure the number of rows is a multiple of 10 in pair.txt'

def valuate(model,
            data_cfg,
            device):

    # feature extractor
    feature_extractor = FeatureExtractor(model)

    # process pairtxt
    test_images_path, pair_list = process_pairtxt(data_cfg['val']['pair_txt'], data_cfg['root'])

    # dataloader
    feature_dataset = PredictImageDatasets(transforms=create_AugTransforms(data_cfg['val']['augment']))
    feature_dataset.imgs_path = test_images_path
    feature_dataloader = DataLoader(feature_dataset, shuffle=False, pin_memory=True, num_workers=data_cfg['nw'],
                                    batch_size=data_cfg['val']['bs'],
                                    collate_fn=PredictImageDatasets.collate_fn)

    evaluator = Evaluator(feature_extractor)
    mean, std = evaluator.test(pair_list, feature_dataloader, device)

    return mean, std
