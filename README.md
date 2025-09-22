# Neural Style Transfer Project

## Overview
This project explores three major Neural Style Transfer (NST) methods:

1. **Gatys et al. (2016)** - Optimization-based NST, high quality but slow.
2. **Johnson et al. (2016)** - Per-style feed-forward NST, real-time speed but fixed style per model.
3. **Huang & Belongie (2017) AdaIN** - Arbitrary style transfer in real-time with Adaptive Instance Normalization.

The project compares these methods in terms of **quality, speed, and flexibility**. A simple and interactive **Streamlit web app** was developed to allow users to try out NST without coding.

---

## Features
- Implementation of three NST methods (Gatys, Johnson, AdaIN).
- Training and inference scripts for Johnson and AdaIN models.
- Pre-trained model checkpoints included (`decoder_final.pth`, Johnson models).
- **Streamlit UI**:
  - Upload content and style images.
  - Choose between methods.
  - Adjust **α slider** for style strength (AdaIN).
  - Download results or preview directly in the app.
- Evaluation using both objective metrics (SSIM, LPIPS, runtime) and subjective survey feedback.

---

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+ (with CUDA if GPU available)
- Torchvision
- NumPy, SciPy, Matplotlib, PIL
- Streamlit

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure
```
.
├── gatys/                # Implementation of Gatys NST
├── johnson/              # Implementation of Johnson feed-forward NST
├── adain/                # Implementation of AdaIN NST
├── checkpoints/          # Pretrained weights (Johnson, AdaIN)
├── ui/                   # Streamlit web app
├── results/              # Example stylized outputs
├── utils/                # Helper functions (losses, data loaders)
└── README.md             # This file
```

---

## Usage

### 1. Run Streamlit App
```bash
streamlit run ui/app.py
```
Upload a content and style image (or trained module for Johnson), pick a method, and get your stylized output.

### 2. Run Gatys NST (Optimization)
```bash
python gatys/run_gatys.py --content content.jpg --style style.jpg --out output.jpg
```

### 3. Run Johnson NST (Feed-forward)
Train a Johnson model:
```bash
python johnson/train.py --dataset /path/to/dataset --epochs 2
```
Stylize an image:
```bash
python johnson/stylize.py --content content.jpg --checkpoint checkpoints/johnson_xxx.pth --out output.jpg
```

### 4. Run AdaIN NST (Arbitrary)
Stylize with AdaIN decoder:
```bash
python adain/stylize.py --content content.jpg --style style.jpg --decoder checkpoints/decoder_final.pth --alpha 0.7 --out output.jpg
```

---

## Evaluation
- **Objective metrics**: SSIM, LPIPS, runtime, memory usage.
- **Subjective evaluation**: small user survey comparing visual quality and control.

---

## Results Summary
- **Gatys**: Best detail, but extremely slow (minutes per image).
- **Johnson**: Real-time, but one model per style.
- **AdaIN**: Real-time + arbitrary style images, good balance of quality and speed.

---

## References
- Gatys, L.A., Ecker, A.S. and Bethge, M., 2016. *Image style transfer using convolutional neural networks*. CVPR.
- Johnson, J., Alahi, A. and Fei-Fei, L., 2016. *Perceptual losses for real-time style transfer and super-resolution*. ECCV.
- Huang, X. and Belongie, S., 2017. *Arbitrary style transfer in real-time with adaptive instance normalization*. ICCV.
- Simonyan, K. and Zisserman, A., 2015. *Very deep convolutional networks for large-scale image recognition*. ICLR.
- Ulyanov, D., Vedaldi, A. and Lempitsky, V., 2016. *Instance normalization: The missing ingredient for fast stylization*. arXiv:1607.08022.
- Paszke, A. et al., 2019. *PyTorch: An imperative style, high-performance deep learning library*. NeurIPS.
- Streamlit Inc., 2019. *Streamlit: The fastest way to build data apps*. https://streamlit.io/
