from PIL import Image

# Load the uploaded image
image_path = '/mnt/data/1.jpg'
image = Image.open(image_path)

# Split the image into its R, G, and B channels
r, g, b = image.split()

# Convert each channel to an image
r_image = Image.merge(
    "RGB", (r, Image.new("L", r.size), Image.new("L", r.size)))
g_image = Image.merge(
    "RGB", (Image.new("L", g.size), g, Image.new("L", g.size)))
b_image = Image.merge(
    "RGB", (Image.new("L", b.size), Image.new("L", b.size), b))

# Save the new images
r_image_path = '/mnt/data/red_channel_image.jpg'
g_image_path = '/mnt/data/green_channel_image.jpg'
b_image_path = '/mnt/data/blue_channel_image.jpg'

r_image.save(r_image_path)
g_image.save(g_image_path)
b_image.save(b_image_path)

(r_image_path, g_image_path, b_image_path)
