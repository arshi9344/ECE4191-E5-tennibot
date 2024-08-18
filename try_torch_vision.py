import torch
import torchvision
import torchvision.transforms as transforms

# First we will set up some transformations to apply to our images
# This just normalises the input images
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# We'll train our model using batches of images, 4 at a time
batch_size = 4

# We'll demonstrate this using an existing dataset CIFAR10, but you can change this to meet your needs
# The first time you run this, it will download the dataset to the ./data folder
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# We'll use a dataloader to feed in random batches of this dataset repeatedly
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# We'll use a test dataset kept separate from our model to validate it, and ensure we aren't overfitting
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# The CIFAR10 dataset has 10 classes we need to tell apart
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')