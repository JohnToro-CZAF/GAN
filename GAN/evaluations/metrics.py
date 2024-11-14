# GAN/evaluations/metrics.py

import torch
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3, vgg16
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

def calculate_fid_score(real_images, generated_images, device):
    """
    Calculates the Fréchet Inception Distance (FID) between real and generated images.
    Args:
        real_images (torch.Tensor): Batch of real images.
        generated_images (torch.Tensor): Batch of generated images.
        device (torch.device): Device to perform computations on.
    Returns:
        float: FID score.
    """
    # Load pre-trained InceptionV3 model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    def get_activations(images):
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            activations = inception(images)
        return activations.cpu().numpy()

    act_real = get_activations(real_images)
    act_fake = get_activations(generated_images)

    # Compute mean and covariance
    mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu_fake, sigma_fake = act_fake.mean(axis=0), np.cov(act_fake, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    # Numerical error might give imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid_score


def calculate_ssim_score(imageA, imageB):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.
    Args:
        imageA (torch.Tensor): Image tensor with shape (C, H, W).
        imageB (torch.Tensor): Image tensor with shape (C, H, W).
    Returns:
        float: SSIM score.
    """
    imageA = imageA.permute(1, 2, 0).cpu().numpy()
    imageB = imageB.permute(1, 2, 0).cpu().numpy()
    ssim_value = ssim(imageA, imageB, multichannel=True, data_range=imageB.max() - imageB.min())
    return ssim_value

def calculate_inception_score(images, device, splits=10):
    """
    Calculates the Inception Score (IS) for generated images.
    """
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # Preprocess images
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    preds = []

    with torch.no_grad():
        for i in range(0, images.size(0), 10):
            batch = images[i:i+10]
            pred = inception_model(batch)
            preds.append(F.softmax(pred, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    split_scores = []

    for k in range(splits):
        part = preds[k * (preds.shape[0] // splits): (k+1) * (preds.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        scores = [entropy(part[i, :], py) for i in range(part.shape[0])]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calculate_perceptual_loss(imageA, imageB, device):
    """
    Calculates the Perceptual Loss (LPIPS) between two images using a pre-trained VGG network.
    """
    vgg = vgg16(pretrained=True).features[:16].to(device).eval()

    with torch.no_grad():
        features_A = vgg(imageA.unsqueeze(0))
        features_B = vgg(imageB.unsqueeze(0))
    loss = F.l1_loss(features_A, features_B)
    return loss.item()
