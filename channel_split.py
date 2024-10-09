import os
from PIL import Image

# Define your paths
# Directory where the original images are stored
input_dir = '/Users/lexoortiz/Desktop/CS/CS307/DL_Final_Project/data/train/boxing/'
# Directory where the new images will be saved
output_dir = '/Users/lexoortiz/Desktop/CS/CS307/DL_Final_Project/data/train/boxing/'

# Iterate through each image
for i in range(1, 117):  # Assuming you have 116 images numbered 001.jpg to 116.jpg
    # Format the image filename with leading zeros
    filename = f'{i:03d}.jpg'
    image_path = os.path.join(input_dir, filename)

    # Check if the image already contains an underscore in the filename
    if '_' in filename:
        print(f"Stopping loop: encountered file with underscore '{filename}'")
        break

    # Load the image
    try:
        image = Image.open(image_path)

        # Check if the image has all three channels (RGB)
        if image.mode != 'RGB':
            print(
                f"Skipping {filename}: Image does not have all 3 RGB channels.")
            continue

        # Split the image into its R, G, and B channels
        r, g, b = image.split()

        # Convert each channel to an image
        r_image = Image.merge(
            "RGB", (r, Image.new("L", r.size), Image.new("L", r.size)))
        g_image = Image.merge(
            "RGB", (Image.new("L", g.size), g, Image.new("L", g.size)))
        b_image = Image.merge(
            "RGB", (Image.new("L", b.size), Image.new("L", b.size), b))

        # Generate new filenames in the format 001_r.jpg, 001_g.jpg, 001_b.jpg
        r_image_path = os.path.join(output_dir, f'{i:03d}_r.jpg')
        g_image_path = os.path.join(output_dir, f'{i:03d}_g.jpg')
        b_image_path = os.path.join(output_dir, f'{i:03d}_b.jpg')

        # Save the images
        r_image.save(r_image_path)
        g_image.save(g_image_path)
        b_image.save(b_image_path)

        print(f"Saved: {r_image_path}, {g_image_path}, {b_image_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
