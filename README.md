# Neural Style Transfer Project

## Overview
This project explores three major Neural Style Transfer (NST) methods:

1. **Gatys et al. (2016)** - Optimization-based NST, high quality but slow.
2. **Johnson et al. (2016)** - Per-style feed-forward NST, real-time speed but fixed style per model.
3. **Huang & Belongie (2017) AdaIN** - Arbitrary style transfer in real-time with Adaptive Instance Normalization.

The project compares these methods in terms of **quality, speed, and flexibility**. A simple and interactive **Streamlit web app** was developed to allow users to try out NST without coding.

---
<img width="500" height="372" alt="Main Outputs" src="https://github.com/user-attachments/assets/a5f7713f-df78-4747-93f8-3682eaaa3082" />

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
├── gatys/                                           # Implementation of Gatys NST
├── johnson/                                         # Implementation of Johnson feed-forward NST
├── adain/                                           # Implementation of AdaIN NST
├── checkpoints/                                     # Pretrained weights (Johnson, AdaIN)
├── results/                                         # Example stylized outputs
├──decoder_final.pth                                 # For the Streamlit web app 
├──requirements.txt                                  # For the Streamlit web app 
├──streamlit_johnson_style_transfer_app 2.py         # Streamlit web app
├──transform_net.pth                                 # For the Streamlit web app 
└── README.md                                        # This file
```

---

##Usage

The easiest way to try the project is through the Streamlit web app:

[🔗 Open the NST App](https://final-project-nst-wyky6a4shqkzp7yp5yp9vs.streamlit.app/)

Upload a content image and a style image (or a trained module for Johnson).

Choose one of the three methods: Johnson or AdaIN.

Adjust parameters (e.g., style strength for AdaIN).

View the stylized output directly and download it if you want.

---

## Evaluation
- **Objective metrics**: SSIM, LPIPS, runtime, memory usage.
- **Subjective evaluation**: small user survey comparing visual quality and control.

---

## Results Summary
- **Gatys**: Best detail, but extremely slow (minutes per image).
- **Johnson**: Real-time, but one model per style.
- **AdaIN**: Real-time + arbitrary style images, good balance of quality and speed.

---<img width="500" height="396" alt="output 2" src="https://github.com/user-attachments/assets/43bfe095-159b-43c8-885f-88ddd36fd3a1" />


## References
- Gatys, L.A., Ecker, A.S. and Bethge, M., 2016. *Image style transfer using convolutional neural networks*. CVPR.
- Johnson, J., Alahi, A. and Fei-Fei, L., 2016. *Perceptual losses for real-time style transfer and super-resolution*. ECCV.
- Huang, X. and Belongie, S., 2017. *Arbitrary style transfer in real-time with adaptive instance normalization*. ICCV.
- Simonyan, K. and Zisserman, A., 2015. *Very deep convolutional networks for large-scale image recognition*. ICLR.
- Ulyanov, D., Vedaldi, A. and Lempitsky, V., 2016. *Instance normalization: The missing ingredient for fast stylization*. arXiv:1607.08022.
- Paszke, A. et al., 2019. *PyTorch: An imperative style, high-performance deep learning library*. NeurIPS.
- Streamlit Inc., 2019. *Streamlit: The fastest way to build data apps*. https://streamlit.io/
