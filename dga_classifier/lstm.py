#!/usr/bin/env python
#encoding=utf-8

"""Train and test LSTM classifier"""
import dga_classifier.data as data
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.cross_validation import train_test_split


def build_model(max_features, maxlen):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    indata = data.get_data()

    # Extract data and labels
    X = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])

    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # Convert labels to 0-1
    y = [0 if x == 'benign' else 1 for x in labels]

    for fold in range(nfolds):
        print("Fold %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, test_size=0.2)

        print('Build model...')
        model = build_model(max_features, maxlen)

        print("Train...")
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=max_epoch,
            validation_data=(X_test, y_test)
        )
        score, acc = model.evaluate(
            X_test, y_test,
            batch_size=batch_size
        )
        print('Test score:', score)
        print('Test accuracy:', acc)

        import datetime
        model.save('lstm.{}.h5'.format(
            datetime.datetime.now().strftime('%Y%M%d.%H%m'))
        )
