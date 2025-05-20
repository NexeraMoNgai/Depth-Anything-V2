import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from nexera_packages.utilities.image_processing import *
from sklearn.cluster import AffinityPropagation, Birch, KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

"""
python detect_flats.py \
    --source "path/to/source" \
    --output "path/to/output" \
    --surfaces 5 \
    --encoder vitl \
    --bandwidth 0.1
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Detect flat surfaces via clustering on normal vectors")
    parser.add_argument('--source', type=Path, default=Path('assets/inputs'), help='Input directory with images')
    parser.add_argument('--output', type=Path, default=Path('assets/outputs'), help='Output directory for results')
    parser.add_argument('--surfaces', type=int, default=5, help='Number of flat surfaces to detect')
    parser.add_argument('--encoder', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'], default='vitl', help='DepthAnythingV2 encoder variant')
    parser.add_argument('--bandwidth', type=float, default=0.1, help='MeanShift bandwidth')
    return parser.parse_args()

def main():
    args = parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    flat_model = {
        "MeanShift": MeanShift(bandwidth=0.1, bin_seeding=True),
        "KMeans": KMeans(n_clusters=args.surfaces, random_state=42),
        "GMM": GaussianMixture(n_components=args.surfaces, covariance_type='full', random_state=42),
        "BIRCH" : Birch(threshold=0.03, n_clusters=args.surfaces),
        # "AffinityPropagation" : AffinityPropagation(damping=0.7) Not efficient enough,
        # "DBSCAN": DBSCAN(eps=0.3, min_samples=100, algorithm='ball_tree') NOT EFFICIENT ENOUGH,
        # "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=50) NOT Efficient enough,
        # "Agglomerative": AgglomerativeClustering(n_clusters=interested_surface) NOT EFFICIENT ENOUGH,
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    # Load depth model
    model = DepthAnythingV2(**model_configs[args.encoder])
    ckpt_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(DEVICE).eval()

    # Prepare output folders
    (args.output / 'normals').mkdir(parents=True, exist_ok=True)
    (args.output / 'depths').mkdir(parents=True, exist_ok=True)
    (args.output / 'flats').mkdir(parents=True, exist_ok=True)

    # Process each image
    for file in tqdm(list(args.source.glob("*")), desc="Processing Images"):
        raw_img = cv2.imread(file)
        if raw_img is None:
            continue  # Skip unreadable files

        dmap = model.infer_image(raw_img)   # H x W
        nmap = dmap_2_nmap(dmap)            # H x W x 3

        name = file.name
        H, W, _ = nmap.shape

        # Save normal and depth maps
        cv2.imwrite(str(args.output / 'normals' / f'normal-{name}'), nmap)
        cv2.imwrite(str(args.output / 'depths' / f'depth-{name}'), dmap)
        
        # Flatten and normalize normal vectors
        normals = nmap.reshape(-1, 3) 
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_normalized = normals / (norms + 1e-8)

        # Dimensionality reduction using PCA
        pca = PCA(n_components=3)
        reduced_normals = pca.fit_transform(normals_normalized)

        for model_name, flat_clustering_model in flat_model.items():
            try:
                labels = flat_clustering_model.fit_predict(reduced_normals)
            except Exception as e:
                print(f"Clustering failed for {model_name} on {name}: {e}")
                continue

            labels_image = labels.reshape(H, W)

            # Find all the clusters
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Generate colored overlay using cmap
            overlay = np.zeros((H, W, 3), dtype=np.uint8)
            cmap = plt.get_cmap('tab20', len(unique_labels))

            for i, label in enumerate(unique_labels):
                if label == -1:
                    continue # skip noise
                color = np.array(cmap(i)[:3]) * 255
                overlay[labels_image == label] = color.astype(np.uint8)

            out_path = args.output / 'flats' / f'{model_name}-{name}'
            cv2.imwrite(str(out_path), overlay)
    
if __name__ == "__main__":
    main()