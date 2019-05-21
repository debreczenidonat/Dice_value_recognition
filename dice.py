# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:52:59 2018

@author: Donát
"""

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from keras.models import Model,load_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,UpSampling2D,BatchNormalization,CuDNNLSTM, multiply, add
from keras.layers.convolutional import Conv2D,AveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras import regularizers

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

##############################################
######check gpu usage
##############################################
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

##############################################
######some function, some of them for using one-time-only
##############################################

#### Read in the different numpy files, and merging them
def read_separate_arrays_to_merge_them(from_id = 100, to_id = 1699, name_of_folder_with_arrays = "./dice/collection_of_arrays_2/"):
    #print(os.listdir("./dice/collection_of_arrays"))
    numpy_arrays_to_merge = os.listdir("./dice/collection_of_arrays_2")
    
    
    #end_bit = np.empty(to_id + 1, dtype = "U4")
    #end_bit[:] = ".npy"
    
    end_bit = np.repeat(".npy",to_id + 1 - from_id)
    id_bit = np.arange(start=from_id,stop=(to_id + 1)).astype('U6')
    numpy_arrays_to_merge = np.core.defchararray.add(id_bit, end_bit)
    
    #np.load("./dice/collection_of_arrays/0.npy").shape
    i = True
    for cur_filename in numpy_arrays_to_merge:
        #print("./dice/collection_of_arrays/" + cur_filename)
        if i:
            collection = np.load(name_of_folder_with_arrays + cur_filename)
            collection = collection.reshape(240,320,3)
            i = False
            continue
        cur_array = np.load(name_of_folder_with_arrays + cur_filename)
        collection = np.concatenate((collection,cur_array),axis = 0)
        
    raw_input = collection.reshape(-1,240,320,3)
    #raw_input[0].shape
    #import matplotlib.pyplot as plt
    #plt.imshow(raw_input[0])
    np.save('all_merged_arrays_2',raw_input)





##############################################
######main branch
##############################################
    
##
#read_separate_arrays_to_merge_them()
    
input_pics = np.load('all_merged_arrays_2.npy')
labels = pd.read_csv('./dice/labels_2.csv', header = None, names = ("cnt","label"))
labels = labels['label'].values
labels = labels[100:]
orig_labels = labels
##to_categorical takes numbers from 0 so... ... update : we have 0 now which means no dice present so we may use 0 again
#labels = labels - 1
labels = to_categorical(labels)
#plt.imshow(input_pics[700])
#labels[700]
train_x, val_x, train_y, val_y = train_test_split(input_pics, labels, test_size=0.2)

##############################################
######model layers
##############################################
#def dense_blocks(dense_block_id = 0, input_layer,  number_of_layers = 3, filter_per_layer = 1, kernel_size_per_layer = 3, block_activation = 'relu'):
#    l = []
#    for i in range(number_of_layers):
#        if i == 0:
#            l[i] = Conv2D(filters = 4, kernel_size = 3 , padding = "same", activation='relu')(input_layer)
#        else:
#            for j in range(i):
#                
#                l[i] = Conv2D(filters = 4, kernel_size = 3 , padding = "same", activation='relu')(l[i-1])
#    conv120 = Conv2D(filters = 4, kernel_size = 3 , padding = "same", activation='relu')(conv000)
#    conv121 = Conv2D(filters = 4, kernel_size = 3 , padding = "same", activation='relu')(conv000)
#    conv122 = Conv2D(filters = 4, kernel_size = 3 , padding = "same", activation='relu')(conv000)

def dense_blocks1(input_layer,num_of_filter = 4):
    l = []
    a = []
    
    l.append(Conv2D(filters = num_of_filter, kernel_size = 3 , padding = "same", activation='relu')(input_layer))
    
    l.append(Conv2D(filters = num_of_filter, kernel_size = 3 , padding = "same", activation='relu')(l[0]))
    a.append(concatenate([l[0],l[1]]))
    
    l.append(Conv2D(filters = num_of_filter, kernel_size = 3 , padding = "same", activation='relu')(a[0]))
    a.append(concatenate([l[0],l[1],l[2]]))
    
    l.append(Conv2D(filters = num_of_filter, kernel_size = 3 , padding = "same", activation='relu')(a[1]))
    a.append(concatenate([l[0],l[1],l[2],l[3]]))
    return  l,a

def chain1(input_layer,filter_list = [4,8]):
    l = []
    a = []
    c = 1
    for i in filter_list:
        if c == 1 :        
            l.append(Conv2D(filters = i, kernel_size = 3 , padding = "same", activation='relu')(input_layer))
            a.append(MaxPooling2D(pool_size=(2, 2))(l[-1]))
        else :
            l.append(Conv2D(filters = i, kernel_size = 3 , padding = "same", activation='relu')(a[-1]))    
            a.append(MaxPooling2D(pool_size=(2, 2))(l[-1]))
        c = c + 1
    return  l,a

def dense1(input_layer):
    d = []

    d.append(Dense(100, activation='sigmoid')(input_layer))
    d.append(Flatten()(d[-1]))
    d.append(Dropout(0.1)(d[-1]))
    d.append(Dense(80, activation='sigmoid')(d[-1]))
    d.append(Dense(10, activation='sigmoid')(d[-1]))
    return  d

def shrink1(input_layer,p_size = 2):
    d =[]
    #d.append(AveragePooling2D(pool_size=(p_size, p_size))(input_layer))
    d.append(MaxPooling2D(pool_size=(p_size, p_size))(input_layer))
    return  d


the_input_layer = Input(shape=(240,320,3))
the_input_layera = BatchNormalization(axis=-1)(the_input_layer)

conv1,pool1 = chain1(the_input_layera,[8,8])
conv2,pool2 = chain1(the_input_layera,[8,8])

merge1 = concatenate([pool1[-1],pool2[-1]] + shrink1(the_input_layera,4))
d1 = Dropout(0.1)(merge1)

conv1m,pool1m = chain1(d1,[32,32])
conv2m,pool2m = chain1(d1,[32,32])

merge2 = concatenate([pool1m[-1],pool2m[-1]] + shrink1(the_input_layera,16))
d2 = Dropout(0.1)(merge2)

conv1mm,pool1mm = chain1(d2,[64,64])
conv2mm,pool2mm = chain1(d2,[64,64])


den1 = dense1(pool1mm[-1])
den2 = dense1(pool2mm[-1])

mer = concatenate([den1[-1],den2[-1]])
output = Dense(7, activation='softmax')(mer)# hidden4


dice_model = Model(inputs=the_input_layer, outputs=output)
# summarize layers
print(dice_model.summary())



##############################################
######Keras model run
##############################################
dice_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False), loss='categorical_crossentropy')#'adam'

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15, verbose=1, mode='auto')

filepath="./saved_keras_models/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose = 0, save_best_only=True, mode='auto')
#callbacks_list = [checkpoint]

history = dice_model.fit(train_x, train_y, epochs = 999, batch_size = 100,validation_data=(val_x, val_y),callbacks=[early_stopping,checkpoint]) #
#dice_model.predict(val_x[0:1,:,:,:])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
##############################################
######confusion matrix
##############################################

saved_dice_model = load_model("./saved_keras_models/weights-improvement-43-1.91.hdf5")
print(saved_dice_model.summary())


#pred_t = saved_dice_model.predict(np.expand_dims(val_x[0], axis = 0))

pred_t = saved_dice_model.predict(val_x)
pred_t = np.argmax(pred_t,axis = 1)
fact_t = np.argmax(val_y,axis = 1)



print(confusion_matrix(pred_t, fact_t))
print(classification_report(pred_t, val_y))

pred_t != fact_t 

pred_t[297]
fact_t [297]
plt.imshow(val_x[0])




##############################################
######Inspecting layers
##############################################
import quiver_engine

quiver_engine.server.launch(saved_dice_model,input_folder = './dice/collection_of_arrays_2')


from keras import backend as K
    from scipy.misc import imsave

layer_dict = dict([(layer.name, layer) for layer in saved_dice_model.layers])
layer_name = 'conv2d_83'
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
#    x *= 255
#    x = x.transpose((1, 2, 0))
#    x = np.clip(x, 0, 255).astype('uint8')
    return x

def show_pic_of_layer(layer_name,filter_index):
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, saved_dice_model.input)[0]
    
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([saved_dice_model.input], [loss, grads])
    
    
    # we start from a gray image with some noise
    #input_img_data = np.random.random((1, 320, 240,3)) * 20 + 128.
    input_img_data = np.expand_dims(val_x[1], axis = 0) *1.
    #input_img_data.shape
    # run gradient ascent for 20 steps
    for i in range(200):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * 1

    
    img = input_img_data[0]
    img = deprocess_image(img)
    
    plt.imshow(img)
    

for layer_nm in layer_dict:
    if layer_nm[0:4] == "conv":
        print(layer_nm)
        for i in range(layer_dict[layer_nm].output_shape[3]) :
            show_pic_of_layer(layer_nm,i)
            plt.draw()
            plt.pause(0.001)

            
layer_dict = dict([(layer.name, layer) for layer in saved_dice_model.layers])        
layer_name = 'conv2d_135'
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer
show_pic_of_layer(layer_name,0)

layer_dict = dict([(layer.name, layer) for layer in saved_dice_model.layers])        
layer_name = 'conv2d_147'

plt.imshow(val_x[1])

layer_name = 'conv2d_156'
for i in range(layer_dict[layer_name].output_shape[3]) :
            show_pic_of_layer(layer_name,i)
            plt.draw()
            plt.pause(0.001)
#imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
            
            
            
            
            
            
            

