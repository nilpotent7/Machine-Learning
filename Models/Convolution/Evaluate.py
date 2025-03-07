import os
import numpy as np
from PIL import Image

sharpen_kernel = np.array([
    -1, -1, -1,
    -1,  9, -1,
    -1, -1, -1
]).reshape(3,3)

blur_kernel = np.array([
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
]).reshape(3,3)

blur_kernel_2 = np.array([
    1, 4, 6, 4, 1, 
    4, 16, 24, 16, 4, 
    6, 24, 36, 24, 6,   
    4, 16, 24, 16, 4,   
    1, 4, 6, 4, 1
]).reshape(5,5)

def FindFiles(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def LoadImage(Path):
    img = Image.open(Path).convert('RGB')
    return (np.array(img, dtype=np.float64))

def PrintImage(Input, SavePath):
    image_array = np.clip(Input, 0, 255)
    img = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
    img.save(SavePath)

class Convolution(object):
    def __init__(self, function, size):
        self.size = size
        self.action = function

    def Perform(self, image):
        half_size = self.size // 2
        padded = np.pad(image, pad_width=((half_size, half_size), (half_size, half_size), (0, 0)), mode='edge')
        output = np.empty_like(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    output[y, x, c] = self.action(padded[y:y+self.size, x:x+self.size, c])
        return output

def Sharpen(array):
    global sharpen_kernel
    return np.sum(array*sharpen_kernel)

def Blur(array):
    global blur_kernel_2
    return np.sum(array*blur_kernel_2) / 256


Input = LoadImage("Input.png")
# SharpenConvolution = Convolution(Sharpen, 3)
BlurConvolution = Convolution(Blur, 5)
Output = BlurConvolution.Perform(Input)
PrintImage(Output, "Output.png")