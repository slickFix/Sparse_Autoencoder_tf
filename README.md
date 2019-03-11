# Sparse_Autoencoder_tf
Comparing performance of two NN models :-
* 1)Sparse autoencoder trained weights, fc model 
* 2)Random initialized weights, fc model.

Dataset used:- MNIST

### 1)Sparse auto encoder trained weights:
* Sparse auto encoder is trained whose encoding layer acts as the 1st layer of the NN
* Then a fully connected NN is trained with the 1st layer as the auto encoder weights
* The auto encoder network architecture is [784,200,784]
* Fully connected NN architecture is [784,200,10]

### 2) Random initialized weights, fc model
* He weight initialisation is used.
* Network architecture of is this fully connected model is [784,200,10]

Using these 2 networks, classification accuracy for the unlabeled data was calculated and **1st model i.e. the model with Sparse auto encoder performed better.** 

#### Code run
* Clone project
* run 'download_mnist.sh'
* run 'sparse_ae.py'

