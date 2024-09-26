import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.manifold import TSNE
from PIL import Image
import os
import shutil
import time
from tqdm import tqdm

# Load image paths and embeddings
embeddings_path = []
X = []

for npy in tqdm(glob.glob("/home/duke/data/favie/v4-embedding/features/*.npy")[:1000]):
    basename = os.path.basename(npy).replace('.npy', '.jpg')
    if os.path.isfile(f"/home/duke/data/favie/v4-embedding/images/{basename}"):
        x = np.load(npy)
        X.append(x)
        embeddings_path.append(npy)
X = np.stack(X)
embeddings_path = np.array(embeddings_path)

# Perform DBSCAN clustering
db = DBSCAN(eps=0.4, 
            min_samples=5, 
            metric="cosine",
            n_jobs=16).fit(X)
# db = HDBSCAN(min_cluster_size = 10,
#              min_samples = 5,
#              cluster_selection_epsilon = 0.2,
#              metric = "cosine",
#              n_jobs = 16).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise (-1 is noise)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

image_src = "/home/duke/data/favie/v4-embedding/images"
cluster = "/home/duke/data/favie/v4-embedding/cluster"
os.makedirs(cluster, exist_ok=True)
for vis_label in range(n_clusters_):
    vis_idx = np.where(labels == vis_label)[0]
    vis_path = embeddings_path[vis_idx]
    vis_path = list(map(lambda x: os.path.basename(x).split(".")[0], vis_path))
    
    for path in vis_path:
        target_dir = os.path.join(cluster, str(vis_label))
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(os.path.join(image_src, f"{path}.jpg"), target_dir)
        
# for path in vis_path:
#     img = Image.open(f"./images-test/{path}.jpg")
#     img = pad_to_square(img)
#     img = resize_image(img)
#     plt.imshow(img)
#     plt.show()
