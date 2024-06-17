import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
import logging as log
log.basicConfig(level=log.INFO)
import argparse

from dataset import load_cifar10_dataloader, load_cifar100_dataloader, load_mnist_dataloader
from models import get_model
from lr import CyclicCosineLRScheduler

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--parent_epochs", type=int, default=200)
    parser.add_argument("--child_epochs", type=int, default=50)
    parser.add_argument("--num_ensembles", type=int, default=10)
    parser.add_argument("--loss_type", type=str, default="temp_ce")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda_kd", type=float, default=0.5)
    parser.add_argument("--scheduler_type", type=str, default="cyclic_cosine")
    parser.add_argument("--max_lr", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--proj_name", type=str, default="distil_ensemble")
    parser.add_argument("--save_dir", type=str, default="models")

    args = parser.parse_args()
    return args

def train(args):
    log.info(f"Starting training with args: {args}")
    
    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    
    wandb.init(project=args.proj_name)
    
    log.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "cifar10":
        train_loader, test_loader = load_cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        train_loader, test_loader = load_cifar100_dataloader(args)
    elif args.dataset == "mnist":
        train_loader, test_loader = load_mnist_dataloader(args)
    else:
        raise ValueError("Invalid dataset")
    num_iters_per_epoch = len(train_loader)
    num_iters_total = num_iters_per_epoch * args.parent_epochs
    log.info(f"Number of iterations per epoch: {num_iters_per_epoch}")
    log.info(f"Total number of iterations: {num_iters_total}")
    log.info(f"Dataset loaded")
    
    log.info(f"Creating model: {args.model}")
    model_p = get_model(args)
    model_p = model_p.to(device)
    
    model_c_list = []
    for i in range(args.num_ensembles):
        model_c = get_model(args)
        model_c = model_c.to(device)
        model_c_list.append(model_c)
    log.info(f"Model created")
    
    log.info(f"Creating optimizer: {args.optimizer}")
    if args.optimizer == "sgd":
        optimizer_p = optim.SGD(model_p.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, momentum=args.momentum)
        optimizer_c_list = [optim.SGD(model_c.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, momentum=args.momentum) for model_c in model_c_list]
    else:
        raise ValueError("Invalid optimizer")
    
    log.info(f"Creating scheduler: {args.scheduler_type}")
    if args.scheduler_type == "cyclic_cosine":
        scheduler_p = CyclicCosineLRScheduler(optimizer_p, args.max_lr, args.min_lr, num_iters_total)
        scheduler_c_list = [CyclicCosineLRScheduler(optimizer_c, args.max_lr, args.min_lr, num_iters_total) for optimizer_c in optimizer_c_list]
    else:
        raise ValueError("Invalid scheduler")
    
    log.info("Starting Parent Model Training...")
    loss_fn = nn.CrossEntropyLoss()
    model_p.train()
    for epoch in range(args.parent_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer_p.zero_grad()
            outputs = model_p(inputs) # (B, num_classes)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.repeat_interleave(outputs.size(1))).mean()
            loss.backward()
            optimizer_p.step()
            scheduler_p.step()
            
            if i % args.log_interval == 0:
                log.info(f"Epoch: {epoch}/{args.parent_epochs}, Iter: {i}/{num_iters_per_epoch}, Loss: {loss.item()}")
                wandb.log({"loss": loss.item()})
    log.info("Parent Model Training finished")
    
    save_path = os.path.join(f"{wandb.run.dir}", f"{args.dataset}_{args.model}_parent.pth")
    torch.save(model_p.state_dict(), save_path)
    log.info(f"Parenet Model saved to {save_path}")
    
    log.info("Starting Child Model Training...")
    for i, model_c in enumerate(model_c_list):
        log.info(f"Training Child Model {i}...")
        model_c.train()
        for epoch in range(args.child_epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer_c_list[i].zero_grad()
                outputs_p = model_p(inputs)  # (B, num_classes)
                outputs_c = model_c(inputs)  # (B, num_classes)
                loss_ce = loss_fn(outputs_c, targets)
                loss_kd = nn.KLDivLoss()(nn.functional.log_softmax(outputs_c/args.temperature, dim=1), nn.functional.softmax(outputs_p/args.temperature, dim=1))
                loss = args.lambda_kd * loss_kd + (1 - args.lambda_kd) * loss_ce
                loss.backward()
                optimizer_c_list[i].step()
                scheduler_c_list[i].step()
                
                if i % args.log_interval == 0:
                    log.info(f"Epoch: {epoch}/{args.child_epochs}, Iter: {i}/{num_iters_per_epoch}, Loss: {loss.item()}")
                    wandb.log({"loss": loss.item()})
        log.info(f"Child Model {i} Training finished")
        
        save_path = os.path.join(f"{wandb.run.dir}", f"{args.dataset}_{args.model}_child_{i}.pth")
        torch.save(model_c.state_dict(), save_path)
        log.info(f"Child Model {i} saved to {save_path}")

if __name__ == "__main__":
    args = get_args()
    train(args)
    

