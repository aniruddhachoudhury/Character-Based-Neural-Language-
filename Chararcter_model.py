# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 20:02:11 2019

@author: Aniruddha choudhury
"""
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load text
raw_text = load_doc('shakespeare.txt')
print(raw_text)

# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)


# organize into sequences of characters
length = 12
sequences = list()
for i in range(length, len(raw_text)):
    # select sequence of tokens
    seq = raw_text[i-length:i+1]
    # store
    sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    
    
# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()    
    # save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)



# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')


chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)


sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]


sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

X.shape

# define the model
def define_model(X):
    model = Sequential()
    model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(vocab_size, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
   # plot_model(model, to_file='model.png', show_shapes=True)
    return model

# define model
model = define_model(X)
# fit model
model.fit(X, y, epochs=100, verbose=2)
# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))