#from signal import SIG_DFL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import utils, datasets, layers, models ### importing from tensorflow.keras and not from keras directly avoids errors 
from sklearn.metrics import  f1_score

from data_loader import (MITBIHDataLoader, PTBDataLoader, DataLoaderUtil,
                            DATA_MITBIH, DATA_PTBDB, DataLoaderUtilMini)


class CNN_layer(tf.keras.layers.Layer):
    def __init__(self,filters, kernel_size,activation1,pool_size,name_="cnn_layer",input_shape_x="None",dropout_layer=False,dropout_value=0.1,first_layer=False):
        super(CNN_layer,self).__init__(name=name_)
        self.dropout_wanted=dropout_layer
        ##############TODO remove activation in conv1D
        ################maybe remove if else if input_shape = None works
        if first_layer:
            self.conv1d=tf.keras.layers.Conv1D(filters, kernel_size, activation=activation1,input_shape=input_shape_x)
        else:
            self.conv1d=tf.keras.layers.Conv1D(filters, kernel_size, activation=activation1)
        self.relu=tf.keras.layers.ReLU()
        self.max=tf.keras.layers.MaxPooling1D(pool_size=pool_size)
        self.dropout= tf.keras.layers.Dropout(dropout_value)
    def call(self, x_data): 
        convo=self.conv1d(x_data)
        convo=self.relu(convo)
        convo=self.max(convo)
        if self.dropout_wanted:
            convo=self.dropout(convo)
        return convo

class CNN_last_layer(tf.keras.layers.Layer):
    def __init__(self,filters, kernel_size,nb_units_dense,activation_dense1,nb_classes,activation_end):
        super(CNN_last_layer,self).__init__(name="cnn_last_layer")
        self.conv1D_last=tf.keras.layers.Conv1D(filters, kernel_size)
        self.ReLU_last = tf.keras.layers.ReLU()
        self.MaxPooling_last= tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(units=nb_units_dense, activation=activation_dense1)
        self.output_layer = tf.keras.layers.Dense(units=nb_classes, activation=activation_end)
    def call(self, x_data):
        convol = self.conv1D_last(x_data)
        convol = self.ReLU_last(convol)
        output_max= self.MaxPooling_last(convol)
        output_dense = self.dense1(output_max)
        output_last = self.output_layer(output_dense)
        return output_last


class CNN_architecture(tf.keras.Model):
    def __init__(self,filter1,kernel1,activation1,pool_size1,input_shape,filter2,kernel2,activation2,pool_size2,filter3,kernel3,activation3,pool_size3,filter4, kernel4,activation4,pool_size4,filter_out,kernel_out,nb_units_dense,activation_dense1,nb_classes,activation_end):
        super(CNN_architecture, self).__init__(name="CNN")
        self.layer1 = CNN_layer(filter1, kernel1,activation1,pool_size1,name_="cnn_layer1",input_shape_x=input_shape,dropout_layer=False,first_layer=True)
        self.layer2 =CNN_layer(filter2, kernel2,activation2,pool_size2,name_="cnn_layer2",dropout_layer=False,first_layer=False)
        self.layer3 = CNN_layer(filter3, kernel3,activation3,pool_size3,name_="cnn_layer3",dropout_layer=True,dropout_value=0.3,first_layer=False)
        self.layer4 = CNN_layer(filter4, kernel4,activation4,pool_size4,name_="cnn_layer4",dropout_layer=True,dropout_value=0.5,first_layer=False)
        self.output_layer =CNN_last_layer(filter_out, kernel_out,nb_units_dense,activation_dense1,nb_classes,activation_end)

    def call(self, input):
        output1=self.layer1(input)
        output2=self.layer2(output1)
        output3=self.layer3(output2)
        output4=self.layer4(output3)
        model_cnn=self.output_layer(output4)
        return model_cnn



if __name__ == "__main__":
    dataloader_util = DataLoaderUtil()
    dataset_name = DATA_PTBDB # or DATA_MITBIH
    
    train_loader, val_loader, test_loader \
        = dataloader_util.get_data_loaders(dataset_name, train_batch_size=200, 
        val_batch_size=1, test_batch_size=100, train_shuffle=False,
        val_split=0.1)
    x_train = train_loader.dataset.x
    y_train = train_loader.dataset.y
    x_test = test_loader.dataset.x
    y_test = test_loader.dataset.y
    x_train=tf.convert_to_tensor(x_train.numpy())
    x_test=tf.convert_to_tensor(x_test.numpy())
    y_train=tf.convert_to_tensor(y_train.numpy())
    y_test=tf.convert_to_tensor(y_test.numpy())
    nb_categories=1
    if dataset_name == DATA_MITBIH:
        y_test = tf.keras.utils.to_categorical(y_test)
        y_train = tf.keras.utils.to_categorical(y_train)
        nb_categories=5
    x_train=tf.transpose(x_train, perm=[0, 2, 1])
    x_test=tf.transpose(x_test, perm=[0, 2, 1])
    if dataset_name == DATA_MITBIH:
        CNN_network = CNN_architecture(filter1=32,kernel1=10,activation1='relu',pool_size1=2,input_shape=(187,1),filter2=32,kernel2=7,activation2='relu',pool_size2=2,filter3=32, kernel3=6,activation3='relu',pool_size3=2,filter4=32, kernel4=3,activation4="relu",pool_size4=2, filter_out=32,kernel_out=3,nb_units_dense=64,activation_dense1="relu",nb_classes=nb_categories,activation_end="softmax")
        epochs_=70
    else:
        CNN_network = CNN_architecture(filter1=32,kernel1=10,activation1='relu',pool_size1=2,input_shape=(187,1),filter2=64,kernel2=7,activation2='relu',pool_size2=2,filter3=128, kernel3=6,activation3='relu',pool_size3=2,filter4=128, kernel4=3,activation4="relu",pool_size4=2, filter_out=64,kernel_out=3,nb_units_dense=128,activation_dense1="relu",nb_classes=nb_categories,activation_end="sigmoid")
        epochs_=70
    #CNN_network_output=CNN_network(x_train) # IF YOU CALL THIS LINE BEFORE THE BUILD THE IMPLMENTATION CRASHES
    if dataset_name == DATA_MITBIH:
        cnn_loss_ce = tf.keras.losses.CategoricalCrossentropy()
        cnn_accuracy=tf.keras.metrics.CategoricalAccuracy()
    else:
        cnn_loss_ce = tf.keras.losses.BinaryCrossentropy()
        cnn_accuracy=tf.keras.metrics.BinaryAccuracy()
    cnn_auroc=tf.keras.metrics.AUC(num_thresholds=200, curve='ROC' ,summation_method='interpolation')
    cnn_auprc=tf.keras.metrics.AUC(num_thresholds=200, curve='PR' ,summation_method='interpolation')
    optimizer_ = tf.keras.optimizers.Adam(learning_rate=1e-2)
    
    CNN_network.build((None,187,1))
    CNN_network.compile(optimizer = optimizer_,loss = cnn_loss_ce,metrics = [cnn_accuracy,cnn_auroc,cnn_auprc])
    #keras.utils.plot_model(CNN_network, show_shapes=True)
    CNN_network.fit(x_train,y_train,batch_size=32,epochs=epochs_,steps_per_epoch=60)
    CNN_network.summary()
    metrics_model = CNN_network.evaluate(x_test,y_test, steps=20,verbose=0)
    print("test loss, test acc:", metrics_model)
    predictions_model = CNN_network.predict(x_test, steps=1)
    if dataset_name == DATA_MITBIH:
        predictions_model = np.argmax(predictions_model, axis=-1)
    else:
        predictions_model = (predictions_model>0.5)
    #f1 = f1_score(y_test, predictions_model[0:21892],labels=np.array([0,1, 2, 3, 4]), average="macro")
    f1 = f1_score(test_loader.dataset.y, predictions_model, average="macro")
    print("Test f1 score : %s "% f1)
    print("model predictions", predictions_model)