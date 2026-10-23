#!/bin/sh
# Downloads weights + the WIDER FACE val parquet (365 MB, deleted after a 160-image subset is extracted).
set -e
cd "$(dirname "$0")"
mkdir -p models data/raw
curl -sL -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sL -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
curl -sL -o data/raw/wider_val.parquet "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
.venv/bin/python -I src/extract_wider.py data/raw/wider_val.parquet data/wider_subset
rm -rf data/raw
