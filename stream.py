#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:10:37 2018

@author: Donát
"""

import io
import socket
import struct
from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Model,load_model

saved_dice_model = load_model("./saved_keras_models/weights-improvement-151-0.03.hdf5")
#print(saved_dice_model.summary())

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')

# for drawing
x =[0,1,2,3,4,5,6]
y = [0,0,0,0,0,0,0]

plt.rcParams.update({'font.size': 15})
plt.ion()
fig,ax = plt.subplots(1,1)
plt.show()

try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        
        
        image = Image.open(image_stream)
        
        np_image = np.array(image)
        np_image = np.expand_dims(np_image, axis = 0)
        #plt.imshow(np_image)
        pred_t = saved_dice_model.predict(np_image)
        #pred_t = np.argmax(pred_t)
        #plt.bar(x_line,x_line,pred_t[0])
        ax.clear()
        ax.bar(x,pred_t[0],align='center')
        ax.relim()
        ax.autoscale_view(True,True,True)
        plt.draw()
        #
        plt.pause(0.001)
        #print(pred_t + 1)
        
        
        
        
        #imageII = image.load()
        #print('Image is %dx%d' % image.size)
        #image.verify()
        #print('Image is verified')
finally:
    connection.close()
    server_socket.close()
    
    
#image.getdata()
#np_image = np.array(image)
#print(image)
#imageII.show()
#img2 = mpimg.imread(image)
