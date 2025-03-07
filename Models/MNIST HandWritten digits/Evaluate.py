import os
import struct
import numpy as np
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk, ImageFilter
from array import array

def on_canvas_update(image):
    image = image.resize((28,28))
    image = np.array(image, dtype=np.float64)
    image = image.reshape((28*28, 1)) / 255
    os.system("cls")
    OutputResult(net.evaluate(image, True), c)

def create_brush_stamp(size=8, blur_radius=2):
    stamp = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(stamp)
    d.ellipse((0, 0, size - 1, size - 1), fill=255)
    stamp = stamp.filter(ImageFilter.GaussianBlur(blur_radius))
    return stamp

def main():
    root = tk.Tk()
    root.title("Zoomed Drawing Canvas with Blurred Brush")
    image_size = 28
    zoom = 10
    canvas_width = image_size * zoom
    canvas_height = image_size * zoom
    image = Image.new("L", (image_size, image_size), "black")
    brush_stamp = create_brush_stamp(size=3, blur_radius=2)
    tk_image = ImageTk.PhotoImage(image.resize((canvas_width, canvas_height), Image.NEAREST))
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="black")
    canvas.pack()
    canvas_img = canvas.create_image(0, 0, anchor="nw", image=tk_image)
    pen_color = "white"
    last_x, last_y = None, None

    def stamp_point(x, y):
        half = brush_stamp.size[0] // 2
        top_left = (x - half, y - half)
        image.paste(pen_color, top_left, mask=brush_stamp)

    def stamp_line(x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        distance = int(max(abs(dx), abs(dy)))
        if distance == 0:
            stamp_point(x0, y0)
        else:
            for i in range(distance + 1):
                x = int(x0 + dx * i / distance)
                y = int(y0 + dy * i / distance)
                stamp_point(x, y)

    def update_canvas():
        nonlocal tk_image
        resized = image.resize((canvas_width, canvas_height), Image.NEAREST)
        tk_image = ImageTk.PhotoImage(resized)
        canvas.itemconfig(canvas_img, image=tk_image)
        on_canvas_update(image)

    def on_button_press(event):
        nonlocal last_x, last_y
        last_x = event.x // zoom
        last_y = event.y // zoom
        stamp_point(last_x, last_y)
        update_canvas()

    def on_move(event):
        nonlocal last_x, last_y
        new_x = event.x // zoom
        new_y = event.y // zoom
        if last_x is not None and last_y is not None:
            stamp_line(last_x, last_y, new_x, new_y)
            last_x, last_y = new_x, new_y
            update_canvas()

    def on_button_release(event):
        nonlocal last_x, last_y
        last_x, last_y = None, None

    def clear_canvas():
        nonlocal image
        image.paste("black", (0, 0, image_size, image_size))
        update_canvas()

    def toggle_pen_color():
        nonlocal pen_color
        pen_color = "white" if pen_color == "black" else "black"
        print(f"Pen color changed to {pen_color}")

    # Bind mouse events to the canvas.
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_button_release)

    # Create a frame for buttons.
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    clear_button = tk.Button(button_frame, text="Clear Canvas", command=clear_canvas)
    clear_button.pack(side="left", padx=5)

    toggle_button = tk.Button(button_frame, text="Swap Pen Color", command=toggle_pen_color)
    toggle_button.pack(side="left", padx=5)

    root.mainloop()


def softmax(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def FindFiles(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        
        return images, labels
        
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.position = 0

    def evaluate(self, a, rounded):
        if(a.shape != (self.sizes[0],1)):
            raise Exception("Input array shape does not correspond to neurons in input layer.")
        
        for b, w in zip(self.biases, self.weights):
            a = softmax(np.dot(w, a)+b)
        if rounded: o = np.round(a, decimals=4) * 100
        else: o = a
        return o

    def LoadDataVar(self, b, w):
        self.weights = w
        self.biases = b
    
    def LoadData(self, path):
        allFiles = FindFiles(path)
        weightsFiles = [x for x in allFiles if "weights" in x]
        biasesFiles = [x for x in allFiles if "biases" in x]
        self.weights = [np.load(w) for w in weightsFiles]
        self.biases = [np.load(b) for b in biasesFiles]

def CalculateCost(Net, TestingSet, DesiredTSet):
    SumOfSum = 0
    Total = 0
    
    for k,x in enumerate(TestingSet):
        desired = DesiredTSet[k]
        result = Net.evaluate(x, False)
        
        overall_sum = 0
        for x,y in zip(result, desired):
            overall_sum += ((x-y)*(x-y))
        
        SumOfSum += overall_sum
        Total +=1

    return SumOfSum / Total

def ConvertDesiredIntoNeurons(Output):
    neurons = []
    for o in Output:
        match o:
            case 0:
                neurons.append(np.array([1,0,0,0,0,0,0,0,0,0]).reshape(10,1))
            case 1:
                neurons.append(np.array([0,1,0,0,0,0,0,0,0,0]).reshape(10,1))
            case 2:
                neurons.append(np.array([0,0,1,0,0,0,0,0,0,0]).reshape(10,1))
            case 3:
                neurons.append(np.array([0,0,0,1,0,0,0,0,0,0]).reshape(10,1))
            case 4:
                neurons.append(np.array([0,0,0,0,1,0,0,0,0,0]).reshape(10,1))
            case 5:
                neurons.append(np.array([0,0,0,0,0,1,0,0,0,0]).reshape(10,1))
            case 6:
                neurons.append(np.array([0,0,0,0,0,0,1,0,0,0]).reshape(10,1))
            case 7:
                neurons.append(np.array([0,0,0,0,0,0,0,1,0,0]).reshape(10,1))
            case 8:
                neurons.append(np.array([0,0,0,0,0,0,0,0,1,0]).reshape(10,1))
            case 9:
                neurons.append(np.array([0,0,0,0,0,0,0,0,0,1]).reshape(10,1))
    return neurons

def LoadImageInput(Path):
    img = Image.open(Path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float64)
    img_array = img_array.reshape((28*28, 1)) / 255
    return img_array

def PrintImage(Input, SavePath):
    img_array = Input.reshape((28, 28))
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(SavePath)

def OutputResult(output, characterMap):
    ot = []
    for k,o in enumerate(output):
        if(len(ot) == 0 or (ot[0][1] <= o and o >= 75)): ot = [(k, o)]
        elif(o >= 1): ot.append((k, o))

    for _,b in enumerate(ot):
        print(f"{b[0]}: {b[1]}")

def ReadImage(fileName):
    im = Image.open(fileName, 'r')
    return [x[0] / 255 for x in list(im.getdata())]


net = Network([784, 64, 64, 10])
net.LoadData("Weights")
c = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# MnistLoader = MnistDataloader("Dataset\\train-images.idx3-ubyte",
#                               "Dataset\\train-labels.idx1-ubyte",
#                               "Dataset\\t10k-images.idx3-ubyte",
#                               "Dataset\\t10k-labels.idx1-ubyte")
# Data = MnistLoader.load_data()
# Inputs = [np.array(x).reshape(784,1) / 255 for x in Data[1][0]]
# DesiredOutputs = ConvertDesiredIntoNeurons(Data[1][1])
# print("Model Error: " + str(CalculateCost(net, Inputs, DesiredOutputs)))

Input = LoadImageInput("Input.png")
# OutputResult(net.evaluate(Input, True), c)

main()