import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from nexera_packages.utilities.image_processing import *
from sklearn.cluster import MeanShift
from tqdm import tqdm


def main():
    source_dir = r'C:\Users\Yankee\Documents\nexera\nexera-core\Depth-Anything-V2\assets\inputs'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
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

        cv2.imwrite(f'outputs/normals/normal-{name}', nmap)
        cv2.imwrite(f'outputs/depths/depth-{name}', dmap)

        # Reshape normal map for clustering
        H, W, _ = nmap.shape
        normals = nmap.reshape(-1, 3) 

        # Normalize normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_normalized = normals / (norms + 1e-8)

        # Mean Shift clustering on normal vectors
        ms = MeanShift(bandwidth=0.1, bin_seeding=True)
        labels = ms.fit_predict(normals_normalized)
        labels_image = labels.reshape(H, W)

        # Find the largest cluster (most common label)
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]

        # Create a mask for the largest flat region
        flat_mask = (labels_image == largest_label).astype(np.uint8)

        # Create a red overlay (RGB image)
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[flat_mask == 1] = [0, 0, 255]  # BGR

        cv2.imwrite(f'outputs/flats/flat-{name}', overlay)
    
if __name__ == "__main__":
    main()