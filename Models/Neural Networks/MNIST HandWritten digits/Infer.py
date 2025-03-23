import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, ImageFilter

from Model import NeuralNetwork

#region Canvas
def on_canvas_update(image):
    image = image.resize((28,28))
    image = np.array(image, dtype=np.float64)
    image = image.reshape((28*28, 1)) / 255
    os.system("cls")
    DecodeOneHot(net.evaluate(image, True), c)

def create_brush_stamp(size=8, blur_radius=2):
    stamp = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(stamp)
    d.ellipse((0, 0, size - 1, size - 1), fill=255)
    stamp = stamp.filter(ImageFilter.GaussianBlur(blur_radius))
    return stamp

def CreateCanvas():
    root = tk.Tk()
    root.title("Canvas")
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
#endregion

def LoadImageInput(Path):
    img = Image.open(Path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float64)
    img_array = ZScoreStandardize(img_array.reshape((28*28, 1)))
    return img_array

# Decode One-Hot neurons and apply labels
def DecodeOneHot(output, characterMap):
    for k,o in enumerate(output):
        print(f"{characterMap[k]} : {o.item():.2f}")

def CalculateAccuracy(Network, Test, Desired):
    correct = 0
    total = 0
    
    for x, desired in zip(Test, Desired):
        output = Network.Evaluate(x)
        
        predicted = np.argmax(output)
        true_label = np.argmax(desired)
        
        if predicted == true_label: correct += 1
        total += 1
        
    return correct / total

# Entire Network's Cost Function: Mean Squared Error
def MeanSquaredError(Network, Test, Desired):
    SumOfSum = 0
    Total = 0
    
    for k,x in enumerate(Test):
        desired = Desired[k]
        result = Network.Evaluate(x)

        overall_sum = 0
        for x,y in zip(result, desired):
            overall_sum += ((x-y)*(x-y))
        
        SumOfSum += overall_sum
        Total += 1

    return SumOfSum / Total

# Entire Network's Cost Function: Cross Entropy Loss
def CrossEntropyLoss(Network, Test, Desired):
    total_cost = 0.0
    total = 0
    epsilon = 1e-12
    
    for k, x in enumerate(Test):
        desired = Desired[k]
        result = Network.Evaluate(x)
        
        cost = -np.sum(desired * np.log(result + epsilon))
        total_cost += cost
        total += 1
        
    return total_cost / total

# Perform Z-Score Standardization on Input
def ZScoreStandardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized_data = (data - mean) / std
    return standardized_data

# Perform Minimum Maximum Range Standardization on Input
def MinMaxStandardize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    range_val = data_max - data_min
    if range_val == 0: return np.zeros_like(data)
    return (data - data_min) / range_val

net = NeuralNetwork([784, 64, 64, 10], NeuralNetwork.Sigmoid, NeuralNetwork.Softmax, NeuralNetwork.CrossEntropyLoss)
net.LoadData("Weights")
c = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

Input = LoadImageInput("Input.png")
DecodeOneHot(net.Evaluate(Input), c)

# CreateCanvas()
