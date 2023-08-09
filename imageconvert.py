import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Load the credit card fraud dataset
df = pd.read_csv('creditcard.csv')

# Create images of transaction amount against time
for index, row in df.iterrows():
    filename = f"{index}.png"
    img = Image.new('RGB', (224, 224), color='white')
    pixels = img.load()
    for i in range(224):
        for j in range(224):
            pixel = int(df.iloc[index, j % 28] * 255)
            pixels[i, j] = (pixel, pixel, pixel)
    img.save(filename)