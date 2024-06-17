from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from typing import Any

def load_cifar10_dataset(args):
    if "efficientnet" in args.model:
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.RandomHorizontalFlip(0.5),
        #     transforms.RandomRotation((-15, 15)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])
        def show_image_size(image, message="Image size"):
            print(f"{message}: {image.size}")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # lambda x: show_image_size(x, "After Resize") or x,
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation((-15, 15)),
            transforms.ToTensor(),
            # lambda x: show_image_size(x, "After ToTensor") or x,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation((-15, 15)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def load_cifar100_dataset(args):
    if "efficientnet" in args.model:
        transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def load_mnist_dataset(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def load_cifar10_dataloader(args: Any):
    train_dataset, test_dataset = load_cifar10_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader

def load_cifar100_dataloader(args: Any):
    train_dataset, test_dataset = load_cifar100_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader

def load_mnist_dataloader(args: Any):
    train_dataset, test_dataset = load_mnist_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    train_dataset, test_dataset = load_cifar10_dataset(args)
