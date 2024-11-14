# Artistic Style Transfer with GANs

This project implements an artistic style transfer system using Generative Adversarial Networks (GANs), including multiple architectures such as CycleGAN, Pix2Pix, and StarGAN. It allows for experimentation with different datasets, models, augmentation techniques, and evaluation metrics.

## **Supported Models**

- **CycleGAN**: Unpaired image-to-image translation.
- **Pix2Pix**: Paired image-to-image translation.
- **StarGAN**: Multi-domain image-to-image translation.

## **Augmentation Techniques**

- **Basic**: Resizing and normalization.
- **Advanced**: Includes flips, rotations, and color jitter.
- **Extra**: Additional augmentations like vertical flips, random grayscale, and Gaussian blur.

## **Evaluation Metrics**

- **Fréchet Inception Distance (FID)**
- **Structural Similarity Index (SSIM)**
- **Inception Score (IS)**
- **Perceptual Loss (LPIPS)**

## **Usage**

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
