import os
import logging as log
log.basicConfig(level=log.INFO)
import argparse
import torch
import numpy as np

from dataset import load_cifar10_dataloader, load_cifar100_dataloader, load_mnist_dataloader
from models import get_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="convnet")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_ensembles", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_single", action="store_true", default=False)
    return parser.parse_args()

def evaluate(args):
    device = torch.device(args.device)
    
    log.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "cifar10":
        train_loader, test_loader = load_cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        train_loader, test_loader = load_cifar100_dataloader(args)
    elif args.dataset == "mnist":
        train_loader, test_loader = load_mnist_dataloader(args)
    else:
        raise ValueError("Invalid dataset")
    
    num_classes = 10 if args.dataset == "cifar10" or args.dataset == "mnist" else 100
    
    log.info(f"Loading model: {args.model}")
    ensembles = []
    for i in range(args.num_ensembles):
        model = get_model(args)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{args.dataset}_{args.model}_child_{i}.pth")))
        model.to(device)
        model.eval()
        ensembles.append(model)
    log.info("Model loaded")
    
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = []
            for j in range(args.num_ensembles):
                outputs.append(ensembles[j](inputs))  # (B, C)
            predicted = torch.mean(torch.stack(outputs, dim=0), dim=0)  # (B, C)
            _, predicted = torch.max(predicted, 1) # (B, )
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        whole_accuracy = correct / total
    
    log.info(f"Test Accuracy: {whole_accuracy}")
    
    if args.eval_single:
        with torch.no_grad():
            correct_list = [0] * args.num_ensembles
            total_list = [0] * args.num_ensembles
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = []
                for j in range(args.num_ensembles):
                    outputs.append(ensembles[j](inputs))  # (B, C)
                outputs = torch.stack(outputs, dim=0)  # (M, B, C)
                for j, out in enumerate(outputs.split(1, dim=0)):
                    out = out.squeeze(0)  # (B, C)
                    _, predicted = torch.max(out, 1)
                    total_list[j] += labels.size(0)
                    correct_list[j] += (predicted == labels).sum().item()
    
        acc_list = [correct / total for correct, total in zip(correct_list, total_list)]    
        for i, acc in enumerate(acc_list):
            log.info(f"Test Accuracy Single for Ensemble[{i}]: {acc}")

if __name__ == "__main__":
    args = get_args()
    evaluate(args)