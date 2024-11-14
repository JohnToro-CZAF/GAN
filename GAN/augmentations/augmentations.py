# GAN/augmentations/augmentations.py

from torchvision import transforms

def get_basic_transforms(img_size):
    """Basic image transformations."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_advanced_transforms(img_size):
    """Advanced image transformations with augmentations."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.12)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_extra_transforms(img_size):
    """Additional transformations."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_augmentation_by_name(aug_name, img_size):
    if aug_name.lower() == 'basic':
        return get_basic_transforms(img_size)
    elif aug_name.lower() == 'advanced':
        return get_advanced_transforms(img_size)
    elif aug_name.lower() == 'extra':
        return get_extra_transforms(img_size)
    else:
        raise ValueError(f"Augmentation {aug_name} not recognized.")