import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

# variated value
lr = 0.05
resolution = 256
epochs = 20

# fixed value
bs = 128
train_ratio = 0.8

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the data transforms
transform_train = transforms.Compose([
    transforms.Resize(resolution),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(degrees=(-15,15)),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Load the CIFAR-10 dataset
data_path = '/home/bumsu/Python/CVclass/data'
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train) 

# define size train and validation dataset
train_size = int(len(trainset) * train_ratio)
val_size = len(trainset) - train_size

# Define training function 
def train_one_epoch(trainloader, model, optimizer, criterion): 
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Move the input data to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # compute train accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # compute train loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    acc = correct / total
    loss = running_loss / i
    return acc, loss

def evaluate_one_epoch(valloader, model, criterion): 
    model.eval()
    
    # validate the model
    correct = 0
    total = 0
    val_loss = 0
    itr = 0
    with torch.no_grad():
        for data in valloader:
            itr += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            outputs = model(inputs)
            
            # compute train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # compute train loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    acc = correct / total
    loss = val_loss / itr

    return acc, loss

def performance_graph (train_acc, train_loss, val_acc, val_loss, seed):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:red')
    ax1.plot(train_acc, color='tab:red', label='train_acc')
    ax1.plot(val_acc, color='tab:red', linestyle='--', label='val_acc')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:blue')
    ax2.plot(train_loss, color='tab:blue', label='train_loss')
    ax1.plot(val_loss, color='tab:blue', linestyle='--', label='val_loss')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='right')

    plt.savefig(f'./graphs/performance_graph_{seed}')

for seed_number in range(1):
    # fix random seed
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

    # Define the ResNet-50 model with pre-trained weights
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = model.to(device)  # Move the model to the GPU

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # Define list of loss for every epoch
    train_loss_epoch = []
    train_acc_epoch = []
    val_loss_epoch = []
    val_acc_epoch = []

    for epoch in np.arange(epochs):
        train_acc, train_loss = train_one_epoch(trainloader, model, optimizer, criterion)
        val_acc, val_loss = evaluate_one_epoch(valloader, model, criterion)
        print(f'[epoch {epoch}] train_acc: {train_acc*100:.2f}%, train_loss: {train_loss:.2f}, val_acc: {val_acc*100:.2f}%, val_loss: {val_loss:.2f}')
        train_acc_epoch.append(train_acc)
        train_loss_epoch.append(train_loss)
        val_acc_epoch.append(val_acc)
        val_loss_epoch.append(val_loss)

    print('Finished Training')

    # Save the checkpoint of the last model
    PATH = f'./models/model_augmentation_{seed_number}'
    torch.save(model.state_dict(), PATH)

    performance_graph(train_acc_epoch, train_loss_epoch, val_acc_epoch, val_loss_epoch, seed_number)
