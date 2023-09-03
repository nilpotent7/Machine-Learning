from PIL import ImageFont, ImageDraw, Image
import numpy as np
import random
import cv2
import os

b,g,r,a = 255,255,255,0

def FindFiles(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# Define math symbols
math_symbols = ["$", "x", "%", "@", "∑", "∞"]

# Find all fonts
fonts = FindFiles("..//Fonts")

# Create a directory to save the images
if not os.path.exists('Symbols\\Testing'):
    os.makedirs('Symbols\\Testing')

for symbol in math_symbols:
    for k,fontpath in enumerate(fonts):
        for version in range(10): # number of variations
            img = np.zeros((50,50,3),np.uint8)

            x = random.randint(5, 25)
            y = random.randint(0, 20)
            position = (x, y)
            symbol_size = 25

            font = ImageFont.truetype(fontpath, symbol_size)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            
            draw.text(position,  symbol, font = font, fill = (b, g, r, a))

            image = np.array(img_pil)

            image_path = os.path.join('Symbols\\Testing', f'∞_{k}_{version}.png')
            cv2.imwrite(image_path, image)

print("Dataset with variations generated successfully.")