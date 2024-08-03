from dataset.basedataset import CBIRDatasets
from dataset.transforms import create_AugTransforms
from dataset.dataprocessor import SmartDataProcessor
from models.faceX.face_model import FaceModelLoader
from engine.vision_engine import yaml_load
from models.faceX.face_model import FeatureExtractor
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import faiss
from typing import Optional

logger = logging.getLogger(__name__)

device = torch.device('cuda:0')

cfgs = yaml_load('/root/autodl-tmp/visiondk/configs/faceX/face.yaml')
is_load_default = True

transforms = create_AugTransforms(cfgs['data']['val']['augment'])
query_dataset, gallery_dataset = CBIRDatasets.build(root='/root/autodl-tmp/visiondk/facedata', transforms=transforms)

query_dataloader = SmartDataProcessor.set_dataloader(query_dataset, bs=256, nw=4, shuffle=False)
gallery_dataloader = SmartDataProcessor.set_dataloader(gallery_dataset, bs=256, nw=4, shuffle=False)

model_loader = FaceModelLoader(model_cfg=cfgs['model'])
model = model_loader.load_weight(model_path='/root/autodl-tmp/visiondk/facedata/Resnet152-irse.pt', ema=False)
feature_extractor = FeatureExtractor(model)

# query_features = feature_extractor.extract_cbir(query_dataloader, device)

def index(extractor: FeatureExtractor, 
          gallery_dataloader: DataLoader, 
          device: torch.device,
          feat_dim: Optional[int] = None,
          dtype: torch.dtype = torch.float16,
          index_factory: str = "Flat", 
          save_path: Optional[str] = None, 
          load_embedding: bool = False,
          ):

    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """

    if load_embedding:
        gallery_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, feat_dim)
    
    else:
        gallery_embeddings = extractor.extract_cbir(gallery_dataloader, device)
        dim = gallery_embeddings.shape[-1]
        
        if save_path is not None:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
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

def evaluate(preds, 
             preds_scores, 
             labels, 
             cutoffs=[1, 3, 10, 20]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
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
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall

    # AUC 
    pred_hard_encodings = []
    for pred, label in zip(preds, labels):
        pred_hard_encoding = np.isin(pred, label).astype(int).tolist()
        pred_hard_encodings.append(pred_hard_encoding)
    
    from sklearn.metrics import roc_auc_score, ndcg_score
    auc = 0
    for idx in range(len(preds_scores)):
        try:
            cur_auc = roc_auc_score(pred_hard_encodings[idx], preds_scores[idx])
        except ValueError:
            cur_auc = 0
        auc += cur_auc
    auc /= len(preds_scores)
    
    metrics[f'AUC@{cutoffs[-1]}'] = auc

    # nDCG
    for k, cutoff in enumerate(cutoffs):
        nDCG = ndcg_score(pred_hard_encodings, preds_scores, k=cutoff)
        metrics[f"nDCG@{cutoff}"] = nDCG
            
    return metrics

faiss_index = index(
    extractor=feature_extractor, 
    gallery_dataloader=gallery_dataloader,
    device=device
    )

scores, indices = search(
    extractor=feature_extractor,
    query_dataloader=query_dataloader,
    faiss_index=faiss_index, 
    k=10,
    )

retrieval_results = []
for indice in indices:
    # filter invalid indices
    indice = indice[indice != -1].tolist()
    retrieval_results.append(gallery_dataset.gallery[indice]['gallery'])

ground_truths = []
for pos in query_dataset.data['pos']:
    ground_truths.append(pos)

metrics = evaluate(retrieval_results, scores, ground_truths)
print(metrics)