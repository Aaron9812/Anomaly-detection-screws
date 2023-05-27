from PIL import Image
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.transforms import ToTensor, Resize, Grayscale, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter

import plotly.offline as pyo
# Set notebook mode to work in
pyo.init_notebook_mode()


def display_image_grid(image_paths):
    num_images = len(image_paths)
    num_rows = (num_images + 1) // 2  # Calculate the number of rows needed for the grid

    # Create the subplots
    fig, axes = plt.subplots(num_rows, 2)

    # Flatten the axes array to make it easier to iterate over
    axes = axes.flatten()

    # Iterate over the images and display them in the subplots
    for i in range(num_images):
        image = Image.open(image_paths[i])
        axes[i].imshow(image, cmap='gray', interpolation='nearest')  # Specify colormap as 'gray' and interpolation as 'nearest'
        axes[i].axis("off")
        image.close()

    # Remove any unused subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()


def check_image_dimensions(directory):
    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if len(image_files) == 0:
        print("No image files found in the directory.")
        return

    # Read the dimensions of the first image
    first_image_path = os.path.join(directory, image_files[0])
    first_image = Image.open(first_image_path)
    first_image_dimensions = first_image.size

    # Iterate over the remaining images and compare their dimensions
    for i in range(1, len(image_files)):
        image_path = os.path.join(directory, image_files[i])
        image = Image.open(image_path)
        image_dimensions = image.size

        if image_dimensions != first_image_dimensions:
            print("Image dimensions are not the same.")
            print("First image dimensions:", first_image_dimensions)
            print("Image path:", image_path)
            return

        image.close()

    print("All images have the same dimensions:", first_image_dimensions)

    first_image.close()


def get_class_encodings(model, test_dataset, device):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_encodings = [[] for i in range(6)]
    with torch.no_grad():
        for data in test_loader:
            img, label = data
            img = Variable(img).to(device)
            encoding = model.encoder(img)
            class_encodings[label.item()].append(encoding.cpu().numpy().ravel())
    return class_encodings

def calculate_class_stats(class_encodings):
    class_means = []
    class_mses = []
    for i in range(len(class_encodings)):
        class_mean = np.mean(class_encodings[i])
        class_mse = np.mean((class_encodings[i] - class_mean) ** 2)
        class_means.append(class_mean)
        class_mses.append(class_mse)
    return class_means, class_mses

def plot_class_stats(class_means, class_mses, class_names):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=np.arange(len(class_means)) - 0.2,
        y=class_means,
        width=0.4,
        name='Mean',
        marker=dict(color='rgba(0, 123, 255, 0.5)')
    ))

    fig.add_trace(go.Bar(
        x=np.arange(len(class_mses)) + 0.2,
        y=class_mses,
        width=0.4,
        name='MSE',
        marker=dict(color='rgba(255, 0, 123, 0.5)')
    ))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(class_names))),
            ticktext=class_names,
            tickangle=90
        ),
        barmode='group',
        title="Class Encodings Mean and MSE Comparison",
        legend=dict(
            x=0,
            y=1.1,
            orientation='h'
        ),
        bargap=0.2,
        bargroupgap=0.1,
        height=600,
        width=800
    )

    fig.show()


def plot_loss(loss_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_values, mode='lines', name='Train Loss'))
    fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss (Log Scale)', yaxis_type='log')
    fig.show()



def get_regular_train_loader(image_size, batch_size):
    transform = transforms.Compose([
        Resize(image_size),  # resize images
        Grayscale(),  # convert to grayscale
        ToTensor(),  # convert to tensor
    ])

    # load all images in the 'good' directory, assuming they're all good
    train_data = ImageFolder(root='screw_data/train', transform=transform)

    # create a DataLoader to handle batching of images
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

def get_test_image_folder(image_size):
    transform = transforms.Compose([
        Resize(image_size),  # resize images
        Grayscale(),  # convert to grayscale
        ToTensor(),  # convert to tensor
    ])
    return ImageFolder(root='screw_data/test', transform=transform)