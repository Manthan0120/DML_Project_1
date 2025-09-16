from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

def get_cifar10(
    root=".",
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    use_normalize=True,
    use_augment=True,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
    shuffle_train=True,
):
    """Load CIFAR-10 dataset with specified transformations and return data loaders."""
    tfms_train = []
    if use_augment:
        tfms_train += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    tfms_train += [transforms.ToTensor()]
    if use_normalize:
        tfms_train += [transforms.Normalize(mean, std)]
    train_tf = transforms.Compose(tfms_train)

    tfms_valid = [transforms.ToTensor()]
    if use_normalize:
        tfms_valid += [transforms.Normalize(mean, std)]
    valid_tf = transforms.Compose(tfms_valid)

    train_data = CIFAR10(root=root, download=True, train=True, transform=train_tf)
    valid_data = CIFAR10(root=root, download=True, train=False, transform=valid_tf)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    
    return train_data, valid_data, train_loader, valid_loader
