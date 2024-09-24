import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from PIL import Image
import os
import shutil


# Load image paths and embeddings
embeddings_path = []
X = []

for npy in glob.glob("/home/duke/data/favie/v2-cluster/features/*.npy"):
    basename = os.path.basename(npy).replace('.npy', '.jpg')
    if os.path.isfile(f"/home/duke/data/favie/v2-cluster/images_cluster_v1/{basename}"):
        x = np.load(npy)
        X.append(x)
        embeddings_path.append(npy)
X = np.stack(X)
embeddings_path = np.array(embeddings_path)

# Perform DBSCAN clustering
db = DBSCAN(eps=0.35, min_samples=5, metric="cosine").fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise (-1 is noise)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

image_src = "/home/duke/data/favie/v2-cluster/images_cluster_v1"
cluster = "/home/duke/data/favie/v2-cluster/cluster_v2"
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
