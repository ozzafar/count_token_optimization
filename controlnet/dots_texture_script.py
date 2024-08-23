from PIL import Image, ImageDraw
import random
import math

width, height = 500, 500
max_num_dots = 25
dot_color = 'blue'

# List to store the positions of the dots

def is_valid_position(new_dot, dot_radius, dots):
    for dot in dots:
        dist = math.sqrt((new_dot[0] - dot[0])**2 + (new_dot[1] - dot[1])**2)
        if dist < 2 * dot_radius:
            return False
    return True

def generate_canny_image(num_dots):
    dots = []
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    while len(dots) < num_dots:
        dot_radius = 30 + (25-num_dots)
        x = random.randint(dot_radius, width - dot_radius)
        y = random.randint(dot_radius, height - dot_radius)
        if is_valid_position((x, y), dot_radius, dots):
            dots.append((x, y))
            draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=dot_color)

    # Save the image
    image.save(f'{num_dots}_dots.png')

    # Display the image
    # image.show()

for i in range(1, max_num_dots):
    generate_canny_image(i)