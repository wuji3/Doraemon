from dataset.basedataset import CBIRDatasets
from dataset.transforms import create_AugTransforms
from dataset.dataprocessor import SmartDataProcessor
from engine.vision_engine import yaml_load
from models.faceX.face_model import FeatureExtractor
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import faiss
from typing import Optional
from sklearn.metrics import roc_auc_score, ndcg_score

logger = logging.getLogger(__name__)

class CBIRMetrics:
    def __init__(self, 
                 cutoffs: list[int] = [1,10, 100]):
        _cutoffs = cutoffs.copy()
        self.cutoffs = _cutoffs
        self.metrics = {}

    def compute_mrr(self, 
                    preds: list[list[str]], 
                    labels: list[list[str]]):
        cutoffs = self.cutoffs.copy()
        mrrs = np.zeros(len(cutoffs))

        for pred, label in zip(preds, labels):
            jump = False
            for i, x in enumerate(pred, 1):
                if x in label:
                    for k, cutoff in enumerate(cutoffs):
                        if i <= cutoff:
                            mrrs[k] += 1 / i
                    jump = True
                if jump:
                    break
        mrrs /= len(preds)
        for i, cutoff in enumerate(cutoffs):
            mrr = mrrs[i]
            self.metrics[f"MRR@{cutoff}"] = mrr
        
    def compute_recall(self,
                       preds: list[list[str]],
                       labels: list[list[str]]):
        cutoffs = self.cutoffs.copy()
        recalls = np.zeros(len(cutoffs))
        for pred, label in zip(preds, labels):
            for k, cutoff in enumerate(cutoffs):
                recall = np.intersect1d(label, pred[:cutoff])
                recalls[k] += len(recall) / len(label)
        recalls /= len(preds)
        for i, cutoff in enumerate(cutoffs):
            recall = recalls[i]
            self.metrics[f"Recall@{cutoff}"] = recall
        
    def compute_precision(self,
                       preds: list[list[str]],
                       labels: list[list[str]]):
        cutoffs = self.cutoffs.copy()
        precisions = np.zeros(len(cutoffs))
        cutoffs = self.cutoffs.copy()
        for pred, label in zip(preds, labels):
            for k, cutoff in enumerate(cutoffs):
                precision = np.intersect1d(label, pred[:cutoff])
                precisions[k] += len(precision) / cutoff
        precisions /= len(preds)
        for i, cutoff in enumerate(cutoffs):
            self.metrics[f"Precision@{cutoff}"] = precisions[i]

    def compute_auc(self,
                    preds: list[list[str]],
                    labels: list[list[str]],
                    preds_scores: list[list[float]]):
        pred_hard_encodings = self.encode_pred2hard(preds=preds, labels=labels)
        
        pred_hard_encodings1d = np.asarray(pred_hard_encodings).flatten() 
        preds_scores1d = preds_scores.flatten()
        auc = roc_auc_score(pred_hard_encodings1d, preds_scores1d)
        
        self.metrics[f'AUC@{self.cutoffs[-1]}'] = auc
    
    def compute_ndcg(self,
                     preds: list[list[str]],
                     labels: list[list[str]],
                     preds_scores: list[list[float]]):
        cutoffs = self.cutoffs.copy()
        pred_hard_encodings = self.encode_pred2hard(preds=preds, labels=labels)
        for _, cutoff in enumerate(cutoffs):
            nDCG = ndcg_score(pred_hard_encodings, preds_scores, k=cutoff)
            self.metrics[f"nDCG@{cutoff}"] = nDCG

    def encode_pred2hard(self,
                         preds: list[list[str]],
                         labels: list[list[str]]) -> list[list[int]]:
        pred_hard_encodings = []
        for pred, label in zip(preds, labels):
            pred_hard_encoding = np.isin(pred, label).astype(int).tolist()
            pred_hard_encodings.append(pred_hard_encoding)
        
        return pred_hard_encodings

    def reset(self):
        self.metrics.clear()


def index(extractor: FeatureExtractor, 
        gallery_dataloader: DataLoader, 
        device: torch.device,
        index_factory: str = "Flat", 
        # need memmap
        memmap_feat_dim: Optional[int] = None,
        memmap_dtype: torch.dtype = torch.float16,
        memmap_save_path: Optional[str] = None, 
        memmap_load_embedding: bool = False,
        ):

    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """

    if memmap_load_embedding:
        gallery_embeddings = np.memmap(
            memmap_save_path,
            mode="r",
            dtype=memmap_dtype
        ).reshape(-1, memmap_feat_dim)
    
    else:
        gallery_embeddings = extractor.extract_cbir(gallery_dataloader, device)
        dim = gallery_embeddings.shape[-1]
        
        if memmap_save_path is not None:
            logger.info(f"saving embeddings at {memmap_save_path}...")
            memmap = np.memmap(
                memmap_save_path,
                shape=gallery_embeddings.shape,
                mode="w+",
                dtype=gallery_embeddings.dtype
            )

            length = gallery_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = gallery_embeddings[i: j]
            else:
                memmap[:] = gallery_embeddings
    
    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if device.type == 'cuda':
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    gallery_embeddings = gallery_embeddings.astype(np.float32)
    faiss_index.train(gallery_embeddings)
    faiss_index.add(gallery_embeddings)
    return faiss_index

def search(extractor: FeatureExtractor, 
        query_dataloader: DataLoader, 
        faiss_index: faiss.Index, 
        k:int = 100, 
        batch_size: int = 256):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = extractor.extract_cbir(query_dataloader, device)
    query_size = query_embeddings.shape[0] 
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices

def compute_metrics(preds, 
                    preds_scores, 
                    labels, 
                    metrics = ['mrr', 'precision', 'recall', 'auc', 'ndcg'],
                    cutoffs=[1, 3, 10]):

    metrics_engine = CBIRMetrics(cutoffs=cutoffs)

    for m in metrics:
        if m == 'mrr':
            metrics_engine.compute_mrr(preds=preds, labels=labels)
        elif m == 'precision':
            metrics_engine.compute_precision(preds=preds, labels=labels)
        elif m == 'recall':
            metrics_engine.compute_recall(preds=preds, labels=labels)
        elif m == 'auc':
            metrics_engine.compute_auc(preds=preds, labels=labels, preds_scores=preds_scores)
        elif m == 'ndcg':
            metrics_engine.compute_ndcg(preds=preds, labels=labels, preds_scores=preds_scores)
        else:
            raise ValueError(f'{m} is not supported')

    return metrics_engine.metrics

def valuate(model,
            data_cfg,
            device):
    
    query_dataset, gallery_dataset = CBIRDatasets.build(root=data_cfg['root'], 
                                                        transforms=create_AugTransforms(data_cfg['val']['augment']))
    
    query_dataloader = SmartDataProcessor.set_dataloader(query_dataset, 
                                                         bs=data_cfg['val']['bs'], 
                                                         nw=data_cfg['nw'], 
                                                         shuffle=False) # must be False, otherwise metrics are computed wrong

    gallery_dataloader = SmartDataProcessor.set_dataloader(gallery_dataset, 
                                                           bs=data_cfg['val']['bs'], 
                                                           nw=data_cfg['nw'], 
                                                           shuffle=False) # must be False, otherwise metrics are computed wrong
                                        
    feature_extractor = FeatureExtractor(model)


    faiss_index = index(
        extractor=feature_extractor, 
        gallery_dataloader=gallery_dataloader,
        device=device
        )

    cutoffs = data_cfg['val']['metrics']['cutoffs']
    scores, indices = search(
        extractor=feature_extractor,
        query_dataloader=query_dataloader,
        faiss_index=faiss_index, 
        k = cutoffs[-1],
        batch_size=data_cfg['val']['bs'],
        )

    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(gallery_dataset.gallery[indice]['gallery'])

    ground_truths = []
    for pos in query_dataset.data['pos']:
        ground_truths.append(pos)

    metrics = compute_metrics(retrieval_results, 
                              scores, 
                              ground_truths, 
                              metrics=data_cfg['val']['metrics']['metrics'],
                              cutoffs=cutoffs)

    return metrics

