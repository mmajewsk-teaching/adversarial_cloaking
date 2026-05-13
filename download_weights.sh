#!/bin/bash
# Downloads the official YOLOv2 weights trained on COCO 80 classes.
# The file is ~203 MB.

set -e

DEST="yolov2/yolov2.weights"

if [ -f "$DEST" ]; then
    echo "Weights already present at $DEST — skipping download."
    exit 0
fi

echo "Downloading YOLOv2 COCO weights..."
curl -L --progress-bar \
    "https://pjreddie.com/media/files/yolov2.weights" \
    -o "$DEST"

echo "Done. Weights saved to $DEST"