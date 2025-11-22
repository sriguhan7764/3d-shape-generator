#!/usr/bin/env python3
"""
Script to download and prepare ShapeNet dataset

ShapeNet is available from:
- Official: https://shapenet.org/ (requires registration)
- Kaggle: https://www.kaggle.com/datasets/balraj98/shapenet-dataset
- HuggingFace: https://huggingface.co/datasets/ShapeNet/ShapeNetCore

This script provides instructions and helpers for dataset preparation.
"""

import os
import argparse
import requests
from tqdm import tqdm
import zipfile
import numpy as np


def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def create_sample_dataset(output_dir, num_samples=100, resolution=32):
    """Create a sample synthetic dataset for testing"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {num_samples} synthetic samples...")

    from utils.shape_utils import create_sample_shape

    shape_types = ['cube', 'sphere', 'pyramid']

    for i in tqdm(range(num_samples)):
        shape_type = shape_types[i % len(shape_types)]
        voxel = create_sample_shape(shape_type, resolution)

        # Add some random noise for variation
        noise = np.random.random(voxel.shape) < 0.05
        voxel = np.logical_or(voxel, noise)

        # Save as numpy array
        np.save(os.path.join(output_dir, f'shape_{i:04d}.npy'), voxel)

    print(f"Sample dataset created in {output_dir}")


def download_from_kaggle(output_dir):
    """Download ShapeNet from Kaggle"""
    print("=" * 60)
    print("Downloading from Kaggle")
    print("=" * 60)
    print()
    print("To download from Kaggle, you need to:")
    print("1. Install kaggle CLI: pip install kaggle")
    print("2. Set up API credentials: https://www.kaggle.com/docs/api")
    print("3. Run: kaggle datasets download -d balraj98/shapenet-dataset")
    print()
    print("Alternatively, download manually from:")
    print("https://www.kaggle.com/datasets/balraj98/shapenet-dataset")
    print()


def download_from_huggingface(output_dir):
    """Download ShapeNet from HuggingFace"""
    print("=" * 60)
    print("Downloading from HuggingFace")
    print("=" * 60)
    print()
    print("To download from HuggingFace:")
    print("1. Install datasets: pip install datasets")
    print("2. Use the following code:")
    print()
    print("from datasets import load_dataset")
    print("dataset = load_dataset('ShapeNet/ShapeNetCore')")
    print()
    print("Alternatively, visit:")
    print("https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
    print()


def main():
    parser = argparse.ArgumentParser(description='Download and prepare ShapeNet dataset')

    parser.add_argument('--output_dir', type=str, default='data/shapenet',
                        help='Output directory for dataset')
    parser.add_argument('--source', type=str, default='sample',
                        choices=['sample', 'kaggle', 'huggingface', 'instructions'],
                        help='Dataset source')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of synthetic samples (for sample source)')
    parser.add_argument('--resolution', type=int, default=32,
                        help='Voxel resolution')

    args = parser.parse_args()

    print("=" * 60)
    print("ShapeNet Dataset Downloader")
    print("=" * 60)
    print()

    if args.source == 'sample':
        print("Creating sample synthetic dataset...")
        create_sample_dataset(args.output_dir, args.num_samples, args.resolution)

    elif args.source == 'kaggle':
        download_from_kaggle(args.output_dir)

    elif args.source == 'huggingface':
        download_from_huggingface(args.output_dir)

    elif args.source == 'instructions':
        print("ShapeNet Dataset Sources:")
        print()
        print("1. Official ShapeNet:")
        print("   https://shapenet.org/")
        print("   - Requires registration")
        print("   - Most complete dataset")
        print()
        print("2. Kaggle:")
        print("   https://www.kaggle.com/datasets/balraj98/shapenet-dataset")
        print("   - Pre-processed subsets available")
        print("   - Easy to download with kaggle CLI")
        print()
        print("3. HuggingFace:")
        print("   https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
        print("   - Available via datasets library")
        print("   - Good integration with PyTorch")
        print()
        print("4. Sample (Synthetic):")
        print("   python download_shapenet.py --source sample")
        print("   - Quick start for testing")
        print("   - No download required")
        print()

    print()
    print("Dataset preparation complete!")
    print("You can now train the model with: python train.py")


if __name__ == '__main__':
    main()
