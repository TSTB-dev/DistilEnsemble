import torch
import torch.nn as nn
import torchvision
import timm

from resnet import resnet18

class ConvNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn1(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv3(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(B, -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x            
    
def get_model(args):
    in_channels = 1 if args.dataset == "mnist" else 3
    if args.dataset == "mnist":
        num_classes = 10
    elif args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        raise ValueError("Invalid dataset")
    
    if args.model == "convnet":
        model = ConvNet(in_channels, num_classes)
    elif args.model == "resnet18":
        # model = torchvision.models.resnet18(pretrained=False)
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = resnet18(num_classes=num_classes)
    elif args.model == "resnet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "vgg16":
        model = torchvision.models.vgg16(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif args.model == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    elif args.model == "efficientnet_b1":
        model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=num_classes)
    elif args.model == "efficientnet_b2":
        model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes)
    elif args.model == "efficientnet_b3":
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
    elif args.model == "efficientnet_b4":
        model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    elif args.model == "efficientnet_b5":
        model = timm.create_model("efficientnet_b5", pretrained=False, num_classes=num_classes)
    elif args.model == "efficientnet_b6":
        model = timm.create_model("efficientnet_b6", pretrained=False, num_classes=num_classes)
    elif args.model == "efficientnet_b7":
        model = timm.create_model("efficientnet_b7", pretrained=False, num_classes=num_classes)
    else:
        raise ValueError("Invalid model")
    
    return model

if __name__ == "__main__":
    model = get_model("convnet", "mnist")
    print(model)
        
    
        