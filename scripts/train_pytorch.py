import os
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """
    Sets up standard ImageNet data transforms and ImageFolder loaders.
    Structure expected:
        data_dir/train/class_x/xxx.jpg
        data_dir/val/class_x/xxx.jpg
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'),
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


def build_model(num_classes: int, freeze_backbone: bool = True):
    """
    Constructs a pretrained ResNet-50 backbone with a custom linear classification head.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes)
    )

    return model


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=20, use_amp=True):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == 'cuda'))

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'--> Saved new best checkpoint with Val Acc: {best_acc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Accuracy: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Image Classifier Training")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--unfreeze', action='store_true', help='Fine-tune full backbone')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    if os.path.exists(args.data_dir) and os.path.exists(os.path.join(args.data_dir, 'train')):
        dataloaders, dataset_sizes, class_names = get_data_loaders(args.data_dir, batch_size=args.batch_size)
        num_classes = len(class_names)
        print(f"Classes ({num_classes}): {class_names}")

        model = build_model(num_classes=num_classes, freeze_backbone=not args.unfreeze).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=args.epochs)
    else:
        print(f"Dataset not found at '{args.data_dir}'. Create 'data/train/' and 'data/val/' folders with class subdirectories.")
