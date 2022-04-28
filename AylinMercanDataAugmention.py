# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:35:47 2022

@author: aylin
"""

from keras.preprocessing.image import load_img, ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np


img = load_img(r"C:\Users\aylin\Desktop\kır\WhatsApp Image 2022-04-25 at 22.33.52.jpeg")

print(type(img))
print(img.format)
print(img.mode)
print(img.size)

data = img
samples = np.expand_dims(data, 0)
datagen = ImageDataGenerator(width_shift_range=[-200, 200], height_shift_range=[-200, 200])
it = datagen.flow(samples,batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

datagen = ImageDataGenerator(horizontal_flip=True)
it = datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

datagen = ImageDataGenerator(rotation_range=90)
it = datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
it = datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
it = datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show() 