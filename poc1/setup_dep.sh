#!/bin/bash
# =============================================================
# Script cài đặt thư viện cho dự án vi-infographic-vqa-poc
# Môi trường: Python >= 3.10, GPU RTX 5090 (Blackwell sm_120)
# CUDA: 12.8 | PyTorch: 2.7.0+cu128
# =============================================================

set -e  # Dừng nếu có lỗi

echo "====================================================="
echo " vi-infographic-vqa-poc - Cài đặt môi trường"
echo "====================================================="

# -----------------------------------------------------------
# BƯỚC 1: Cài PyTorch + torchvision từ index CUDA 12.8 (cu128)
# Bắt buộc để support RTX 5090 Blackwell (sm_120)
# -----------------------------------------------------------
echo "[1/3] Cài PyTorch 2.7.0 + torchvision 0.22.0 (CUDA 12.8)..."
pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# -----------------------------------------------------------
# BƯỚC 2: Cài Hugging Face Ecosystem
# - transformers >= 4.50.0: fix ổn định cho Qwen2.5-VL
#   (4.48 hỗ trợ, nhưng 4.49 có breaking change với Qwen2.5-VL;
#    dùng 4.50.x là phiên bản ổn định nhất hiện tại)
# - accelerate: bắt buộc khi dùng device_map="auto"
# - peft: LoRA fine-tuning (POC 2)
# - trl: fine-tuning pipeline (POC 2)
# - bitsandbytes: quantized training (QLoRA)
# -----------------------------------------------------------
echo "[2/3] Cài Hugging Face Ecosystem..."
pip install \
    "transformers==4.57.1" \
    "accelerate==1.6.0" \
    "peft==0.15.2" \
    "trl==0.16.1" \
    "bitsandbytes==0.45.4"

# -----------------------------------------------------------
# BƯỚC 3: Cài thư viện hỗ trợ VLM + Metrics + Utilities
# -----------------------------------------------------------
echo "[3/3] Cài VLM utils, metrics và các thư viện phụ trợ..."
pip install \
    "qwen-vl-utils==0.0.11" \
    "pillow==11.2.1" \
    "python-Levenshtein==0.27.1" \
    "rapidfuzz==3.12.2" \
    "tqdm==4.67.1" \
    "matplotlib==3.10.1" \
    "pyyaml==6.0.2" \
    "timm==1.0.15" \
    "einops==0.8.1"

echo ""
echo "====================================================="
echo " Cài đặt hoàn tất!"
echo " Kiểm tra PyTorch + CUDA:"
echo "   python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))\""
echo "====================================================="