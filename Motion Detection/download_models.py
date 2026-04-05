"""
Download pre-trained MobileNet SSD model files for person detection.

These files are required for the DNN-based person detector used by main.py.
Run this script once before using the motion detection system.

The MobileNet SSD model is trained on PASCAL VOC and can detect 20 object
classes including 'person'. It runs in real-time on CPU via OpenCV's DNN module.
"""

import os
import sys
import urllib.request

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
}

def download_models():
    for filename, url in FILES.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            print(f"Already exists: {filename}")
            continue

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  Downloaded: {size_mb:.1f} MB")
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")
            sys.exit(1)

    print("\nAll model files ready.")

if __name__ == "__main__":
    download_models()
