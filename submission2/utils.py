from PIL import Image
import matplotlib.pyplot as plt
import os

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