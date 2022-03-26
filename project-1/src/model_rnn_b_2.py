from turtle import forward
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import one_hot
from sklearn.metrics import accuracy_score, f1_score
from data_loader import (MITBIHDataLoader, PTBDataLoader, DataLoaderUtil,
                            DATA_MITBIH, DATA_PTBDB, DataLoaderUtilMini)


class RNN(nn.Module):
    def __init__(self, input_size, nb_classes, hidden_size, n_layers):
        super().__init__()
        self.nb_hidden= hidden_size
        self.nb_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True, nonlinearity= 'relu')   
        self.linear_layer = nn.Linear(hidden_size, nb_classes)
    def forward(self,input_tensor):
        #h0 = torch.zeros(self.nb_layers,input_tensor.size(0), self.nb_hidden) #.to(device)
        #hidden_output=self.hidden_layer1(input_tensor.size(0))
        output, hidden_state = self.rnn(input_tensor)
        output = output.reshape(output.shape[0], -1)
        output = self.linear_layer(output)
        return output
    #def hidden_layer1(self, batch_size):
     #   hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
      #  return hidden_state

if __name__ == "__main__":
    dataloader_util = DataLoaderUtil()
    dataset_name = DATA_MITBIH  # DATA_PTBDB # or DATA_MITBIH
    
    train_loader, val_loader, test_loader \
        = dataloader_util.get_data_loaders(dataset_name, train_batch_size=200, 
        val_batch_size=1, test_batch_size=100, train_shuffle=False,
        val_split=0.1)
    x_train = train_loader.dataset.x
    y_train = train_loader.dataset.y
    x_test = test_loader.dataset.x
    y_test = test_loader.dataset.y
    #x_train=np.transpose(x_train, (0, 2, 1))
    #x_test=np.transpose(x_test, (0, 2, 1))
    print(np.shape(x_train))
    print(np.shape(y_train))

    nb_class=np.zeros(5)
    for i in range(1,6):
        nb_class[i-1] = len(np.where(y_train == i-1)[0])
        print("nb in class ", i-1, ": ", nb_class[i-1])

    nb_epochs = 10
    learning_rate=0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 187
    dim_hidden_layer = 50
    nb_layers=20
    nb_classes_= 5
    batch_size = 1
 
    rnn_model = RNN(input_size, nb_classes=nb_classes_, hidden_size=dim_hidden_layer, n_layers=nb_layers)
    loss_funct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)

    seq_dim = 28  
    loss_list = []
    iteration_list = []
    accuracy_list = []
    loss_count = 0
    
    for epoch in range(nb_epochs):
        ## trainning 
        for i in range(x_train.size(0)):
            inputs=x_train[i,:,:]
            label =y_train[i]
            target = torch.tensor(label, dtype=torch.long)
            #label=torch.nn.functional.one_hot(label, num_classes=5) 
            # remove gradients from previous step
            optimizer.zero_grad()
            outputs = rnn_model(inputs)
            #_, output_max = torch.max(outputs.data, 1)
            loss = loss_funct(np.squeeze(outputs), target)
            loss.backward()
            # Update parameters each step
            optimizer.step()
            
            loss_count += loss.item()
            
        average_loss = loss_count / x_train.size(0)
        print ("Epoch: ", epoch, "Average loss: ", average_loss)


nb_samples = 0
acc_total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i in y_test:
        inputs_test=x_test[i,:,:]
        label_test =y_test[i]
        # calculate outputs by running images through the network
        outputs_test = rnn_model(inputs_test)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs_test.data, 1)
        f1 = f1_score(label_test, predicted, average="macro")
        print("Test f1 score : %s "% f1)

        acc = accuracy_score(label_test, predicted)
        print("Test accuracy score : %s "% acc)
        nb_samples += 1
        acc_total += acc
