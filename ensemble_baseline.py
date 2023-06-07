import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm

seed_number = 5

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the data transforms
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Load the CIFAR-10 test dataset
data_path = '/home/bumsu/Python/CVclass/data'
testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

# Define the list of models for ensemble
models = []
for i in range(seed_number):
    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', num_classes=10)
    model.load_state_dict(torch.load(f'./models/model_baseline_{i}'))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the GPU
    models.append(model)

# Evaluate the ensemble of models
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU  
        outputs = torch.zeros(100, 10).to(device)  # Initialize the output tensor with zeros
        
        for model in models:
            model_outputs = model(inputs)
            outputs += model_outputs
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test dataset: {100 * correct / total:.2f}%')
