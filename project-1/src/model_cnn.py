#from signal import SIG_DFL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils, datasets, layers, models ### importing from tensorflow.keras and not from keras directly avoids errors 
from keras.layers import InputLayer
import torch

from data_loader import (MITBIHDataLoader, PTBDataLoader, DataLoaderUtil,
                            DATA_MITBIH, DATA_PTBDB, DataLoaderUtilMini)


#Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
#  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings 

#config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, 
#                        inter_op_parallelism_threads=2, 
#                        allow_soft_placement=True,
#                        device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})
#session = tf.Session(config=config)
class CNN_layer(layers.Layer):
    def __init__(self,filters, kernel_size,activation1,padding_,pool_size,name_="cnn_layer",input_shape_x="None"):
        super(CNN_layer,self).__init__(name="cnn_layer")
        self.conv1D=layers.Conv1D(filters, kernel_size,activation=activation1, padding=padding_,input_shape=input_shape_x)
        self.ReLU = layers.ReLU()
        self.MaxPooling= layers.MaxPooling1D(pool_size)
    
    def call(self, x_data): 
        #input_data=keras.Input(x_data.shape[1:])
        convo = self.conv1D(x_data)
        convo = self.ReLU(convo)
        convo= self.MaxPooling(convo)
        return convo

class CNN_last_layer(layers.Layer):
    def __init__(self,filters, kernel_size,padding_,nb_classes,activation_end):
        super(CNN_last_layer,self).__init__(name="cnn_layer")
        self.conv1D=layers.Conv1D(filters, kernel_size, padding=padding_)
        self.conv1D_2=layers.Conv1D(filters, kernel_size-2, padding=padding_)
        self.dropout_=layers.Dropout(0.2)
        self.ReLU = layers.ReLU()
        self.output_layer = layers.Dense(units=nb_classes, activation=activation_end)
        self.AveragePooling= layers.GlobalAveragePooling1D()
    def call(self, x_data):
        convol = self.conv1D(x_data)
        convol = self.dropout_(convol)
        convol = self.conv1D_2(convol) 
        convol = self.ReLU(convol)
        output_dense = self.output_layer(convol)
        output_last= self.AveragePooling(output_dense)
        return output_last


class CNN_architecture(keras.Model):
    def __init__(self,filter1,kernel1,activation1,filter2,kernel2,activation2,filter3,kernel3,activation3,padding,pool_size,filter_out,kernel_out,nb_classes,activation_end,input_shape_x):
        super(CNN_architecture, self).__init__(name="CNN")
        self.layer1 = CNN_layer(filter1,kernel1,activation1,padding,pool_size,input_shape_x)
        self.layer2 =CNN_layer(filter2,kernel2,activation2,padding,pool_size)
        self.layer3 = CNN_layer(filter3,kernel3,activation3,padding,pool_size)
        self.output_layer =CNN_last_layer(filter_out,kernel_out,padding,nb_classes,activation_end)

    def call(self, input):
        #input_layer=keras.layers.InputLayer(input.shape[1:]) ##############shape to check
        output1=self.layer1(input)
        output2=self.layer2(output1)
        #output3=self.layer3(output2)
        model_cnn=self.output_layer(output2)
        return model_cnn


    
def CNN_model(x_test,y_test,x_train,y_train):
####### TO DO choose the filter size accordingly
# size will be (if batch then batch,input_size - kernel +1, filters)
    CNN_network = CNN_architecture(filter1=30,kernel1=10,activation1='relu',filter2=20,kernel2=3,activation2='relu',filter3=30, kernel3=3,activation3='sigmoid', padding="same",pool_size=2,filter_out=30,kernel_out=15,nb_classes=5,activation_end="softmax",input_shape_x=(187,1))
    CNN_network_output=CNN_network(x_train)

    cnn_loss_ce = keras.losses.CategoricalCrossentropy()
    cnn_accuracy=keras.metrics.CategoricalAccuracy()
    cnn_auroc=keras.metrics.AUC(num_thresholds=200, curve='ROC' ,summation_method='interpolation')
    cnn_auprc=keras.metrics.AUC(num_thresholds=200, curve='PR' ,summation_method='interpolation')
    ########################## ADD f1 score
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    
    CNN_network.compile(optimizer = optimizer,loss = cnn_loss_ce,metrics = [cnn_accuracy,cnn_auroc]) #,cnn_auprc])
    #keras.utils.plot_model(CNN_network, show_shapes=True)
    CNN_network.fit(x_train,y_train,batch_size=32,epochs=5,steps_per_epoch=10)
    CNN_network.summary()
    metrics_model = CNN_network.evaluate(x_test,y_test)
    print("test loss, test acc:", metrics_model)
    predictions_model = CNN_network.predict(x_test)
    print("model predictions", predictions_model)
 


if __name__ == "__main__":
    dataloader_util = DataLoaderUtil()
    dataset_name = DATA_MITBIH  # DATA_PTBDB # or DATA_MITBIH
    
    train_loader, val_loader, test_loader \
        = dataloader_util.get_data_loaders(dataset_name, train_batch_size=200, 
        val_batch_size=1, test_batch_size=100, train_shuffle=False,
        val_split=0.1)
    x_train = train_loader.dataset.x
    y_train = train_loader.dataset.y
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = test_loader.dataset.x
    y_test = test_loader.dataset.y
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train=tf.transpose(x_train, perm=[0, 2, 1])
    x_test=tf.transpose(x_test, perm=[0, 2, 1])
    print(np.shape(x_train))
    print(np.shape(y_train))
    # converting to categorical data ------- if needed
    #y_train= keras.utils.to_categorical(y_train,num_classes=5)
    #y_train=np.array(y_train)

    CNN_model(x_test,tf.convert_to_tensor(y_test),x_train,tf.convert_to_tensor(y_train))
'''
def building_model(input_shape):
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters1=60, kernel_size=15, padding="same")(input_layer)
# conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1= keras.layers.MaxPooling1D(pool_size=2)(conv1)
    print(conv1.shape)
    conv2 = keras.layers.Conv1D(filters2=30, kernel_size=3, padding="same")(conv1)
# conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2= keras.layers.MaxPooling1D(pool_size=2)(conv2)
    conv3 = keras.layers.Conv1D(filters3=30, kernel_size=3, padding="same")(conv2)
    #conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)   
    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes=5, activation="softmax")(gap)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)
  
model_cnn = building_model(input_shape=x_train.shape[1:]) ##############maybe need to for loop around it 
keras.utils.plot_model(model_cnn, show_shapes=True)
# other  : 
model.add(layers.Dense(x,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
#still need to  .compile and .fit and .val at the end
  '''