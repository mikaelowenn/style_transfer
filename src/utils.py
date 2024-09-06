from tkinter import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def plot_results(original, style, result):
    plt.figure(figsize=(12, 4))
    images = [original, style, result]
    titles = ['Original Image', 'Style Image', 'Stylized Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()