import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# Paths to your images
image_path_1 = "../results/heatmap/17_15 oranges_6.0_heatmap.jpg"
image_path_2 = "../results/heatmap/0_5 bottles_1.001953125_heatmap.jpg"

# Captions for the images
caption_1 = 'Top View'
caption_2 = 'Side View'

# Load the images
img1 = mpimg.imread(image_path_1)
img2 = mpimg.imread(image_path_2)

# Create a figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

# Set a border width
border_width = 3
border_color = 'black'

def add_border(ax, border_color, border_width):
    rect = patches.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        linewidth=border_width, edgecolor=border_color, facecolor='none')
    ax.add_patch(rect)

# Display the first image with its caption
axes[0].imshow(img1)
axes[0].set_title(caption_1, fontsize=20)  # Set font size and weight
axes[0].axis('off')  # Hide axes
add_border(axes[0], border_color, border_width)

# Display the second image with its caption
axes[1].imshow(img2)
axes[1].set_title(caption_2, fontsize=20)  # Set font size and weight
axes[1].axis('off')  # Hide axes
add_border(axes[1], border_color, border_width)

# Adjust layout to prevent overlap and add padding
# plt.subplots_adjust(wspace=0.6)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
plt.tight_layout(pad=0.1)

# Show the figure
plt.savefig("heatmap_different_views.pdf")
plt.show()
