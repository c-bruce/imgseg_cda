import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2

def plot_ellipses_on_image(image, ellipse_df, output_path):
    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image[...,::-1], aspect='auto')
    plt.axis('off')

    # Plot each ellipse on top of the image
    for index, row in ellipse_df.iterrows():
        x = row['x']
        y = row['y']
        major_axis = row['majorAxis']
        minor_axis = row['minorAxis']
        angle = row['angle']

        # Create the ellipse patch
        ellipse_patch = Ellipse((x, y), major_axis, minor_axis, angle=angle,
                                edgecolor='springgreen', facecolor='none', linewidth=2)

        # Add the ellipse patch to the plot
        plt.gca().add_patch(ellipse_patch)

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

def plot_three_square_images(image_name):
    image1 = cv2.imread(f"{image_name}.png")
    image2 = cv2.imread(f"{image_name}_segmented.png")
    image3 = cv2.imread(f"{image_name}_ellipses.png")

    # Create a figure with 3 subplots arranged horizontally
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot the first image
    axes[0].imshow(image1[...,::-1], aspect='auto')
    axes[0].axis('off')

    # Plot the second image
    axes[1].imshow(image2[...,::-1], aspect='auto')
    axes[1].axis('off')

    # Plot the third image
    axes[2].imshow(image3[...,::-1], aspect='auto')
    axes[2].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{image_name}_comparison.png", bbox_inches='tight', pad_inches=0)
