import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, utils
import torch
from sklearn.metrics import  f1_score
from data_loader import (MITBIHDataLoader, PTBDataLoader, DataLoaderUtil,
                            DATA_MITBIH, DATA_PTBDB, DataLoaderUtilMini)

class RNN_layers(layers.Layer):
    def __init__(self,input_dim, output_dim,nb_units_rnn, nb_classes,activation_dense,dropout_, input_layer_):
        super(RNN_layers, self).__init__(name="RNN_layers")
        #self.input_layer=layers.Input(shape=(187,1))
        self.simple_rnn = layers.SimpleRNN(nb_units_rnn,dropout=dropout_,input_shape=(187,1))
        self.dropout_=layers.Dropout(0.2)
        self.dense = layers.Dense(units=nb_classes, activation=activation_dense)
        self.input_layer_bool=input_layer_
    def call(self, x_data):
        if self.input_layer_bool:
            #input_ = self.input_layer(x_data)
            rnn = self.simple_rnn(x_data)
        else:
            rnn= self.dropout_(x_data)
        output_rnn=self.dense(rnn)
        return output_rnn


class RNN_architecture(keras.Model):
    def __init__(self,input_dim, output_dim,nb_units_rnn,dropout_,nb_classes,activation_dense,input_dim2, output_dim2,nb_units_rnn2,dropout_2,nb_classes2,activation_dense2):
        super(RNN_architecture, self).__init__(name="RNN")
        self.rnn_layers = RNN_layers(input_dim, output_dim,nb_units_rnn,nb_classes,activation_dense,dropout_,input_layer_=True)
        self.rnn_layers_2 = RNN_layers(input_dim2, output_dim2,nb_units_rnn2,nb_classes2,activation_dense2,dropout_2,input_layer_=False)
    def call(self, input):
        rnn_layer1=self.rnn_layers(input)
        return self.rnn_layers_2(rnn_layer1)
        
def RNN_model(x_test,y_test,x_train,y_train):
    RNN_network = RNN_architecture(input_dim=x_test.shape[1:], output_dim=5,nb_units_rnn=38,nb_classes=5,activation_dense="relu",dropout_=0.1,input_dim2=x_test.shape[1:], output_dim2=5,nb_units_rnn2=30,nb_classes2=5,activation_dense2="linear",dropout_2=0.5)
    RNN_network_=RNN_network(x_train)

    rnn_loss_ce = keras.losses.CategoricalCrossentropy()
    rnn_accuracy=keras.metrics.CategoricalAccuracy()
    rnn_auroc=keras.metrics.AUC(num_thresholds=200, curve='ROC' ,summation_method='interpolation')
    rnn_auprc=keras.metrics.AUC(num_thresholds=200, curve='PR' ,summation_method='interpolation')
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    RNN_network.compile(optimizer = optimizer,loss = rnn_loss_ce,metrics = [rnn_accuracy,rnn_auroc,rnn_auprc])
    tf.keras.utils.plot_model(RNN_network, show_shapes=True)
    RNN_network.fit(x_train,y_train,epochs=1,batch_size=16,steps_per_epoch=2)
    RNN_network.summary()
    
    predictions_model = RNN_network.predict(x_test,steps=20)
    print("model predictions", predictions_model)
    #f1 = f1_score(y_test, predictions_model,labels=np.array([0,1, 2, 3, 4]), average="macro")
    #print("Test f1 score : %s "% f1)
    metrics_model = RNN_network.evaluate(x_test,y_test,steps=20,verbose=1)

    print("test loss, test acc:", metrics_model)

if __name__ == "__main__":
    dataloader_util = DataLoaderUtil()
    dataset_name = DATA_MITBIH  # DATA_PTBDB # or DATA_MITBIH
    
    train_loader, val_loader, test_loader \
        = dataloader_util.get_data_loaders(dataset_name, train_batch_size=200, 
        val_batch_size=1, test_batch_size=100, train_shuffle=False,
        val_split=0.1)
    x_train = train_loader.dataset.x
    #x_train=x_train.numpy()
    y_train = train_loader.dataset.y
    #y_train=y_train.numpy()
    
    x_test = test_loader.dataset.x
    #x_test=x_test.numpy()
    y_test = test_loader.dataset.y
    #y_test=y_test.numpy()
    indexes=np.zeros(5)
    nb_class=np.zeros(5)
    for i in range(1,6):
        nb_class[i-1] = len(np.where(y_train == i)[0])
        print("nb in class ", i-1, ": ", nb_class[i-1])
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train=tf.transpose((x_train), perm=[0, 2, 1])
    x_test=tf.transpose(x_test, perm=[0, 2, 1])
    print(np.shape(x_train))
    print(np.shape(y_train))

    RNN_model(x_test,tf.convert_to_tensor(y_test),x_train,tf.convert_to_tensor(y_train))

