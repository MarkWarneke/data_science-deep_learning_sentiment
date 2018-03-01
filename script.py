import tflearn
from tflearn.data_utils import to_categorical, pad_sequence
from tflearn.datasets import imdb

# IMBD Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)

trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
# Vectorize inputs - Convert to numerical representation
# padd each sequence with a zero at end, which max sequence length is 100
trainX = pad_sequence(trainX, maxlen=100, value=0.)
textX = pad_sequence(testX, maxlen=100, value=0.1)

# convert labels to binary vectors with classes 0 = positve 1 = negative
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Build network
# Input layer, where data is feeded - specify inputs shape. First Element: Batch size to none Length: Equal to Max Sequence Length 
net = tflearn.input_data([None, 100])
# Embedding layer
# use output of previous as input layer
# set to 10.000 because loaded 10.000 words from dataset
# output dim to 128 number of dimension of resulting embedding 
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
# feed LSTM (long short term meomry )
# allow to remeber form beginning of sequences, improve prediction
# dropout technice to prevent overfitting by randomly turning of an on pathways
net = tflearn.lstm(net, 128, dropout=0.8)

# fully connected, every layer in previous is connected to every in this
# adding fully connected layer computationaly cheap way of non linear combination of them
# softmax activiation funtion, take vector and squash it to output propability 0 to 1 (sigmoid)
net = tflearn.fully_connected(net, 2, activation='softmax')
# last layer is regression layer : apply regression operation, 
# optimizer is adam (uses gradient decent) minimize given loss funktion
# and learning rate (how fast network to learn)
# loss categorical crossentroopy helps find difference predicted and expcteed
net = tflearn.regression (net, optimizer='adam', learning_rate=0.0001, loss='categorical_crosstropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
# show_metric view log of accuracy
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)