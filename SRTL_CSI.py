# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:22:44 2020

@author: Heidy
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, ReLU, Dropout,Lambda, LSTM, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio
import numpy as np
import math
import time
import gc
from tensorflow.python.framework import ops




#%%
#nampilin nama dari semua layer bersama wieght dan bias nya dan shape nya juga di checkpoint

from tensorflow.python import pywrap_tensorflow
import os

path = "model_dir/"

checkpoint_path = os.path.join(path,"SRCNN.model-2550000")
#checkpoint_path = os.path.isfile("SRCNN.model-2550000.meta")
#reader = tf.train.NewCheckpointReader(checkpoint_path)
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape) # Remove this is you want to print only variable names
    
  
#%%
    
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# import os
# #checkpoint_path = os.path.isfile("SRCNN.model-2550000.meta")
# path = "model_dir/"

# checkpoint_path = os.path.join(path,"SRCNN.model-2550000")

# # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
# print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='',  all_tensors=True)


#%%

ops.reset_default_graph()
#tf.reset_default_graph()

#envir = 'indoor'  # 'indoor' or 'outdoor'

# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

# network params
residual_num = 1
epnum=100    #epoch number, deafult 1000 from github

#cobaa 3x1 input dan io
def residual_network(x, residual_num, encoded_dim):  # residual network with dropout
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        shortcut = y
        # y = Conv2D(64, kernel_size=(1, 4), padding='same', data_format='channels_first')(y)
        # y = add_common_layers(y)

        # y = Conv2D(16, kernel_size=(1, 8), padding='same', data_format='channels_first')(y)
        # y = add_common_layers(y)
        
        #y = GaussianNoise(0.1)

        y = Conv2D(2, kernel_size=(1, 6), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)
        #y = add_common_layers(y)
        
        y = add([shortcut, y])
        y = LeakyReLU()(y)
        #y =ReLU()(y)
        return y

#-----------------Transfer Learning------------------
    # #buat key nya dulu 
    weight_key1 = 'w1'
    weight_key2 = 'w2'
    weight_key3 = 'w3'
    
    bias_key1 = 'b1'
    bias_key2 = 'b2'
    bias_key3 = 'b3'

    #terus masukin biar bisa dibaca
    weight1 = reader.get_tensor(weight_key1)
    weight2 = reader.get_tensor(weight_key2)
    weight3 = reader.get_tensor(weight_key3)

    bias1 = reader.get_tensor(bias_key1)
    bias2 = reader.get_tensor(bias_key2)
    bias3 = reader.get_tensor(bias_key3)


    def super_resolution_front(y):
        y = Conv2D( 64 , (9 , 9) ,
                    weights=[weight1,bias1],
                #kernel_initializer='he_normal',
                activation = 'relu', 
                padding='same',
                data_format='channels_first')(y)
        y = Conv2D( 32 , (1 , 1) , 
                    weights=[weight2,bias2],
                #kernel_initializer='he_normal',
                activation = 'relu', 
                padding='same',
                data_format='channels_first')(y)
        y = Conv2D( 1 , (5 , 5) , 
                    weights=[weight3,bias3],
                #kernel_initializer='he_normal', 
                padding='same',
                data_format='channels_first')(y)
        return y

    def super_resolution_back(y):
        y = Conv2D( 64 , (9 , 9) ,
                    weights=[weight1,bias1],
                #kernel_initializer='he_normal',
                activation = 'relu', 
                padding='same',
                data_format='channels_first')(y)
        y = Conv2D( 32 , (1 , 1) , 
                    weights=[weight2,bias2],
                #kernel_initializer='he_normal',
                activation = 'relu', 
                padding='same',
                data_format='channels_first')(y)
        y = Conv2D( 1 , (5 , 5) , 
                    weights=[weight3,bias3],
              #kernel_initializer='he_normal', 
                padding='same',
                data_format='channels_first')(y)
        return y

#encoder dropout
    x = Conv2D(2, (1, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    
    x = Reshape((img_total,))(x)

#fully connected layer
    encoded = Dense(encoded_dim, activation='linear')(x)
    encoded = Dropout(0.1)(encoded)

#decoder dropout

#fully connected layer
    x = Dense(img_total, activation='linear')(encoded)
    x = Dropout(0.1)(x)
    # x = GaussianNoise(0.2)(x)
    #x = LSTM(50)(x)

    x = Reshape((img_channels, img_height, img_width,))(x)
    for i in range(residual_num):
        x = residual_block_decoded(x)

#jadi value nya x , nah kan dimensinya 2 jadi di SPLIT kan lan jadi 1,1 dengan menggunakan axis = 1 penjelasan axis ada di buku ccatatan
    x_front,x_back = tf.split(x, [1,1], 1)

# manggil 2 def yg diatas terus dalam kurung manggil (yg sudah di Split kan) 
    x_front = super_resolution_front(x_front)
    #_front.layers.trainable = False
    x_back =  super_resolution_back(x_back)
    #x_back.trainable = False

    #x = tf.stack([x_front, x_back], axis=1) SALAH
    # jadi disini kenapa pake tf.stack salah karena stack itu prinsipnya adalahm MENGGABUNGKAN dengan nambah dimensi +1 (R+1) jadi misal awalnya 2 dimensi jadi 3 dimensi

# maka pakailah tf.concatenate dimana concat itu sama MENAGGABUNGKAN tapi dia prinsipnya meskipun gabung , dimensi tetep sama. hanyaaa panjang ato lebar nya lebih besar
# menggabungkan tanpa meningkatkan dimensi
    x = tf.keras.backend.concatenate((x_front, x_back), axis=1)

    #img_split1 = tf.split(value=x, num_or_size_splits= 2, axis=1) SALAH



    x = Conv2D(2, (1, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)


    return x


#------Prosess compile training dan testingnya-----

image_tensor = Input(shape=(img_channels, img_height, img_width))

#logix = [True, False]
logix = [True]
#compressr= [512, 128, 64, 32]  # compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
compressr= [32]
environ = ['indoor']#environ=['indoor', 'outdoor']

for envir in environ:

  for with_dropout in logix:

    for encoded_dim in compressr:

       if with_dropout:
         network_output = residual_network(image_tensor, residual_num, encoded_dim)

       # elif not with_dropout:
       #   network_output = residual_network(image_tensor, residual_num, encoded_dim)

       autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
       autoencoder.compile(optimizer='adam', loss='mse', metrics=(['acc']))
       print(autoencoder.summary())


       # Data loading
       if envir == 'indoor':
         mat = sio.loadmat('data/DATA_Htrainin.mat')
         x_train = mat['HT'] # array
         mat = sio.loadmat('data/DATA_Hvalin.mat')
         x_val = mat['HT'] # array
         mat = sio.loadmat('data/DATA_Htestin.mat')
         x_test = mat['HT'] # array

      
       del mat
       gc.collect()

       x_train = x_train.astype('float32')
       x_val = x_val.astype('float32')
       x_test = x_test.astype('float32')

       x_train = np.reshape(x_train, ( 
           len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
                             # 2 x 32 x32            
       x_val = np.reshape(x_val, (
           len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
       x_test = np.reshape(x_test, (
           len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format


       class LossHistory(Callback):
         def on_train_begin(self, logs={}):
           self.losses_train = []
           self.losses_val = []
           self.acc_train = []
           self.acc_val = []

         def on_batch_end(self, batch, logs={}):
           self.losses_train.append(logs.get('loss'))
           self.acc_train.append(logs.get('acc'))

         def on_epoch_end(self, epoch, logs={}):
           self.losses_val.append(logs.get('val_loss'))
           self.acc_val.append(logs.get('val_acc'))
           

       history = LossHistory()
       file = 'CsiNet_' + (envir) + '_dim' + str(encoded_dim) + 'dropout' + str(with_dropout)
       path = 'result\TensorBoard_%s' % file

       autoencoder.fit(x_train, x_train,
                       epochs=epnum,  # default 1000 (from github)
                       batch_size=200,  # default 200 (from github)
                       shuffle=True,
                       validation_data=(x_val, x_val),
                       callbacks=[history,
                                  TensorBoard(log_dir=path)])
       # ngelaurin nilai testing
        
       score = autoencoder.evaluate(x_test,x_test, batch_size=200)
       print (score)
       

       
       # Testing data
       tStart = time.time()
       x_hat = autoencoder.predict(x_test)
       tEnd = time.time()
       dec_time = (tEnd - tStart) / x_test.shape[0]

       #save the result
       filename = 'result/trainloss_%s.csv' % file
       loss_history = np.array(history.losses_train)
       np.savetxt(filename, loss_history, delimiter=",")

       filename = 'result/valloss_%s.csv' % file
       loss_history = np.array(history.losses_val)
       np.savetxt(filename, loss_history, delimiter=",")

       filename = 'result/dec_time_%s.csv' % file
       np.savetxt(filename, [dec_time], delimiter=",")
       
       filename = 'result/trainacc_%s.csv' % file
       acc_history = np.array(history.acc_train)
       np.savetxt(filename, acc_history, delimiter=",")

       filename = 'result/valacc_%s.csv' % file
       acc_history = np.array(history.acc_val)
       np.savetxt(filename, acc_history, delimiter=",")  

       print("It cost %f sec" % (dec_time))
       
       # Calcaulating the NMSE and rho
       if envir == 'indoor': 
         mat = sio.loadmat('data/DATA_HtestFin_all.mat')
         X_test = mat['HF_all']# array
       
       X_test = np.reshape(X_test, (len(X_test), img_height, 125))
       x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
       x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
       x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
       x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
       x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
       x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
       x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
       X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
       X_hat = X_hat[:, :, 0:125]
       
       n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))
       n1 = n1.astype('float64')
       n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))
       n2 = n2.astype('float64')
       aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))
       rho = np.mean(aa/(n1*n2), axis=1)
       X_hat = np.reshape(X_hat, (len(X_hat), -1))
       X_test = np.reshape(X_test, (len(X_test), -1))
       power = np.sum(abs(x_test_C)**2, axis=1)
       power_d = np.sum(abs(X_hat)**2, axis=1)
       mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)
       
       print("In "+envir+" environment")
       print("When dimension is", encoded_dim)
       print("NMSE is ", 10*math.log10(np.mean(mse/power)))
       print("Correlation is ", np.mean(rho))
       filename = "result/decoded_%s.csv"%file
       x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
       np.savetxt(filename, x_hat1, delimiter=",")
       filename = "result/rho_%s.csv"%file
       np.savetxt(filename, rho, delimiter=",")
       
       import matplotlib.pyplot as plt
       '''abs'''
       n = 10
       plt.figure(figsize=(20, 4))
       for i in range(n):
         # display origoutal
         ax = plt.subplot(2, n, i + 1 )
         x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
         plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
         plt.gray()
         ax.get_xaxis().set_visible(False)
         ax.get_yaxis().set_visible(False)
         ax.invert_yaxis()

         # display reconstruction
         ax = plt.subplot(2, n, i + 1 + n)
         decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
                          + 1j*(x_hat[i, 1, :, :]-0.5))
         plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
         plt.gray()
         ax.get_xaxis().set_visible(False)
         ax.get_yaxis().set_visible(False)
         ax.invert_yaxis()
       plt.show()

       # save
       # serialize model to JSON
       # model_json = autoencoder.to_json()
       # outfile = "result/model_%s.json"%file
       # with open(outfile, "w") as json_file:
       #     json_file.write(model_json)
       # serialize weights to HDF5
       outfile = "result/model_%s.h5"%file
       autoencoder.save_weights(outfile)


       del x_train
       del x_val
       del x_test
       del x_hat
            
       gc.collect()


#----------------- bagian coba2 --------------------

# def residual_networkwd(x, residual_num, encoded_dim):  # residual network with dropout
#     def add_common_layers(y):
#         y = BatchNormalization()(y)
#         y = LeakyReLU()(y)
#         return y

#     def residual_block_decoded(y):
#         shortcut = y
#         # y = Conv2D(64, kernel_size=(1, 4), padding='same', data_format='channels_first')(y)
#         # y = add_common_layers(y)

#         # y = Conv2D(16, kernel_size=(1, 8), padding='same', data_format='channels_first')(y)
#         # y = add_common_layers(y)
        
#         #y = GaussianNoise(0.1)

#         y = Conv2D(2, kernel_size=(1, 6), padding='same', data_format='channels_first')(y)
#         y = BatchNormalization()(y)
#         #y = add_common_layers(y)
        
#         y = add([shortcut, y])
#         y = LeakyReLU()(y)
#         #y =ReLU()(y)
#         return y

# #-----------------Transfer Learning------------------
#     # #buat key nya dulu 
#     weight_key1 = 'w1'
#     weight_key2 = 'w2'
#     weight_key3 = 'w3'
    
#     bias_key1 = 'b1'
#     bias_key2 = 'b2'
#     bias_key3 = 'b3'

#     #terus masukin biar bisa dibaca
#     weight1 = reader.get_tensor(weight_key1)
#     weight2 = reader.get_tensor(weight_key2)
#     weight3 = reader.get_tensor(weight_key3)

#     bias1 = reader.get_tensor(bias_key1)
#     bias2 = reader.get_tensor(bias_key2)
#     bias3 = reader.get_tensor(bias_key3)


#     def super_resolution_front(y):
#         y = Conv2D( 64 , (9 , 9) ,
#                     weights=[weight1,bias1],
#                 #kernel_initializer='he_normal',
#                 activation = 'relu', 
#                 padding='same',
#                 data_format='channels_first')(y)
#         y = Conv2D( 32 , (1 , 1) , 
#                     weights=[weight2,bias2],
#                 #kernel_initializer='he_normal',
#                 activation = 'relu', 
#                 padding='same',
#                 data_format='channels_first')(y)
#         y = Conv2D( 1 , (5 , 5) , 
#                     weights=[weight3,bias3],
#                 #kernel_initializer='he_normal', 
#                 padding='same',
#                 data_format='channels_first')(y)
#         return y

#     def super_resolution_back(y):
#         y = Conv2D( 64 , (9 , 9) ,
#                     weights=[weight1,bias1],
#                 #kernel_initializer='he_normal',
#                 activation = 'relu', 
#                 padding='same',
#                 data_format='channels_first')(y)
#         y = Conv2D( 32 , (1 , 1) , 
#                     weights=[weight2,bias2],
#                 #kernel_initializer='he_normal',
#                 activation = 'relu', 
#                 padding='same',
#                 data_format='channels_first')(y)
#         y = Conv2D( 1 , (5 , 5) , 
#                     weights=[weight3,bias3],
#               #kernel_initializer='he_normal', 
#                 padding='same',
#                 data_format='channels_first')(y)
#         return y

# #encoder dropout
#     x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
#     x = add_common_layers(x)

    
#     x = Reshape((img_total,))(x)

# #fully connected layer
#     encoded = Dense(encoded_dim, activation='linear')(x)
#     encoded = Dropout(0.1)(encoded)

# #decoder dropout

# #fully connected layer
#     x = Dense(img_total, activation='linear')(encoded)
#     x = Dropout(0.1)(x)
#     # x = GaussianNoise(0.2)(x)
#     #x = LSTM(50)(x)

#     x = Reshape((img_channels, img_height, img_width,))(x)
#     for i in range(residual_num):
#         x = residual_block_decoded(x)

# #jadi value nya x , nah kan dimensinya 2 jadi di SPLIT kan lan jadi 1,1 dengan menggunakan axis = 1 penjelasan axis ada di buku ccatatan
#     x_front,x_back = tf.split(x, [1,1], 1)

# # manggil 2 def yg diatas terus dalam kurung manggil (yg sudah di Split kan) 
#     x_front = super_resolution_front(x_front)
#     #_front.layers.trainable = False
#     x_back =  super_resolution_back(x_back)
#     #x_back.trainable = False

#     #x = tf.stack([x_front, x_back], axis=1) SALAH
#     # jadi disini kenapa pake tf.stack salah karena stack itu prinsipnya adalahm MENGGABUNGKAN dengan nambah dimensi +1 (R+1) jadi misal awalnya 2 dimensi jadi 3 dimensi

# # maka pakailah tf.concatenate dimana concat itu sama MENAGGABUNGKAN tapi dia prinsipnya meskipun gabung , dimensi tetep sama. hanyaaa panjang ato lebar nya lebih besar
# # menggabungkan tanpa meningkatkan dimensi
#     x = tf.keras.backend.concatenate((x_front, x_back), axis=1)

#     #img_split1 = tf.split(value=x, num_or_size_splits= 2, axis=1) SALAH



#     x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)


#     return x


