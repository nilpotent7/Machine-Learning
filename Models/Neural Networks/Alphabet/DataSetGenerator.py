from PIL import ImageFont, ImageDraw, Image
import numpy as np
import random
import cv2
import os
import tqdm

b, g, r, a = 255, 255, 255, 0

def FindFiles(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
alphabets2 = [x.upper() for x in alphabets]
alphabets = [*alphabets, *alphabets2]
fonts = FindFiles("Fonts")

if not os.path.exists('Symbols\\Alphabets'):
    os.makedirs('Symbols\\Alphabets')

img_width, img_height = 50, 50
symbol_size = (30,50)
versions = 25

total_iterations = len(alphabets) * len(fonts) * versions
with tqdm.tqdm(total=total_iterations, desc="Generating Images") as pbar:
    for i, symbol in enumerate(alphabets):
        for k, fontpath in enumerate(fonts):
            for version in range(versions):
                img = np.zeros((img_height, img_width, 3), np.uint8)
                font = ImageFont.truetype(fontpath, random.randrange(symbol_size[0], symbol_size[1]))
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                bbox = draw.textbbox((0, 0), symbol, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (img_width - text_width) / 2 - bbox[0]
                y = (img_height - text_height) / 2 - bbox[1]
                position = (x, y)
                
                draw.text(position, symbol, font=font, fill=(b, g, r, a))
                
                image = np.array(img_pil)
                image_path = os.path.join('Symbols\\Alphabets', f'{i}_{k}_{version}.png')
                cv2.imwrite(image_path, image)
                pbar.update(1)

print("Dataset with variations generated successfully.")
