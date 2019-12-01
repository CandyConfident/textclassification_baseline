from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.preprocessing import sequence

from han_model import HAN

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
max_sents = 16
max_sen_len = 32
batch_size = 64
embedding_dims = 100
epochs = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x #sentence x #word)...')
x_train = sequence.pad_sequences(x_train, maxlen=max_sents * max_sen_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_sents * max_sen_len)
x_train = x_train.reshape((len(x_train), max_sents, max_sen_len))
x_test = x_test.reshape((len(x_test), max_sents, max_sen_len))
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = HAN(max_sents, max_sen_len, max_features, embedding_dims).get_model()

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)