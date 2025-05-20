import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from nexera_packages.utilities.image_processing import *
from sklearn.cluster import DBSCAN, MeanShift
from tqdm import tqdm

# these colors are in BGR format
predefined_colors = [
    [0, 0, 255],    # Red
    [0, 255, 0],    # Green
    [255, 0, 0],    # Blue
    [0, 255, 255],  # Yellow
    [255, 0, 255],  # Magenta
    [255, 255, 0]   # Cyan
]

def main():
    source_dir = r'C:\Users\Yankee\Documents\nexera\nexera-core\Depth-Anything-V2\assets\inputs'
    output_dir = r'C:\Users\Yankee\Documents\nexera\nexera-core\Depth-Anything-V2\assets\outputs'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    interested_surface = 5

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    flat_model = {
        "MeanShift": MeanShift(bandwidth=0.1, bin_seeding=True),
        # "DBSCAN": DBSCAN(eps=0.3, min_samples=100, algorithm='ball_tree'),
        # "KMeans": KMeans(n_clusters=interested_surface, random_state=42),
        # "Agglomerative": AgglomerativeClustering(n_clusters=interested_surface),
        # "GMM": GaussianMixture(n_components=interested_surface, covariance_type='full', random_state=42)
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    files = Path(source_dir).iterdir()
    for file in tqdm(files, desc="Pipeline"):
        raw_img = cv2.imread(file)

        dmap = model.infer_image(raw_img) # HxW raw depth map in numpy

        nmap = dmap_2_nmap(dmap)

        name = file.name

        os.makedirs(f'{output_dir}/normals/', exist_ok = True)
        os.makedirs(f'{output_dir}/depths/', exist_ok = True)
        os.makedirs(f'{output_dir}/flats', exist_ok = True)

        cv2.imwrite(f'{output_dir}/normals/normal-{name}', nmap)
        cv2.imwrite(f'{output_dir}/depths/depth-{name}', dmap)

        # Reshape normal map for clustering
        H, W, _ = nmap.shape
        normals = nmap.reshape(-1, 3) 

        # Normalize normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_normalized = normals / (norms + 1e-8)

        for m_name, flat_clustering_model in flat_model.items():
            labels = flat_clustering_model.fit_predict(normals_normalized)
            labels_image = labels.reshape(H, W)

            # Find all the clusters
            unique_labels, counts = np.unique(labels, return_counts=True)
            sort_counts = np.argsort(-counts)

            overlay = np.zeros((H, W, 3), dtype=np.uint8)

            for i in sort_counts[:interested_surface]:
                interested_label = unique_labels[i]

                flat_mask = (labels_image == interested_label).astype(np.uint8)

                overlay[flat_mask == 1] = predefined_colors[i]

            cv2.imwrite(f'{output_dir}/flats/{m_name}-{name}', overlay)
    
if __name__ == "__main__":
    main()