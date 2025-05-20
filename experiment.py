import argparse
import re
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from depth_anything_v2.dpt import DepthAnythingV2
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
    parser.add_argument('--encoder', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'], default='vitl', help='DepthAnythingV2 encoder variant')
    return parser.parse_args()

def natural_key(path):
    # Extract the filename as string
    filename = path.name
    # Split by digits and convert digit parts to int for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def main():
    args = parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Load depth model
    model = DepthAnythingV2(**model_configs[args.encoder])
    ckpt_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(DEVICE).eval()

    # Prepare output folders
    (args.output / 'depths').mkdir(parents=True, exist_ok=True)

    points = []

    files = sorted(list(args.source.glob("*")), key=natural_key)

    # Process each image
    for file in tqdm(files, desc="Processing Images"):
        raw_img = cv2.imread(file)
        if raw_img is None:
            continue  # Skip unreadable files

        start_time = time.perf_counter()
        for _ in range(10):
            dmap = model.infer_image(raw_img)   # H x W
        
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        points.append(elapsed_time/10)

        print(f'\ndepth-{file.name}-{args.encoder} avg time taken {elapsed_time/10}')

        # Save depth maps
        cv2.imwrite(str(args.output / 'depths' / f'{args.encoder}-{file.name}'), dmap)

    plt.figure(figsize=(10, 5))
    x_vals = list(range(1, len(points) + 1))  # X-axis from 1 to N
    y_vals = points

    # Plot as bars
    plt.bar(x_vals, y_vals, color='blue', label="Avg Time per Image")

    # Annotate each bar with its value
    for x, y in zip(x_vals, y_vals):
        plt.text(x, y + 0.001, f"{y:.3f}", ha='center', fontsize=8)

    plt.ylabel("Average Time Taken (s)")
    plt.xlabel("Test Image Index")
    plt.title(f"{args.encoder} Runtime per Test Image")
    plt.xticks(x_vals)  # Ensure all x-ticks are shown
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()