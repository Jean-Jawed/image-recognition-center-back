#!/usr/bin/env python3
"""
Download MediaPipe Task model files.

This script downloads the required .task model files for:
- Hand Landmarker
- Pose Landmarker  
- Face Landmarker

Run this script before starting the backend server.
"""

import os
import urllib.request
import sys

# Model URLs from Google Storage
MODELS = {
    "hand_landmarker.task": 
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_heavy.task": 
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "face_landmarker.task": 
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

# Target directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"  Downloading: {os.path.basename(dest_path)}")
        print(f"  From: {url}")
        
        # Create a simple progress indicator
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print()  # Newline after progress
        return True
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("MediaPipe Task Models Downloader")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\nModels directory: {MODELS_DIR}")
    
    # Download each model
    success_count = 0
    for filename, url in MODELS.items():
        dest_path = os.path.join(MODELS_DIR, filename)
        
        print(f"\n[{list(MODELS.keys()).index(filename) + 1}/{len(MODELS)}] {filename}")
        
        # Check if already exists
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  Already exists ({size_mb:.1f} MB) - skipping")
            success_count += 1
            continue
        
        # Download
        if download_file(url, dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  ✓ Downloaded successfully ({size_mb:.1f} MB)")
            success_count += 1
        else:
            print(f"  ✗ Download failed")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Downloaded: {success_count}/{len(MODELS)} models")
    
    if success_count == len(MODELS):
        print("✓ All models ready!")
        return 0
    else:
        print("✗ Some downloads failed. Please retry.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
