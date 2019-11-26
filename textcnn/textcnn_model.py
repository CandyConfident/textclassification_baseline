# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, \
    Dropout,GlobalAveragePooling1D

class TextCNN(object):
    def __init__(self, maxlen, input_dim, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid',embedding=None):
        self.maxlen = maxlen
        # int > 0. Size of the vocabulary,
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = embedding

    def get_model(self):

        input = Input((self.maxlen,))
        if not self.embedding:
            emb_layer = Embedding(self.input_dim, self.embedding_dims, input_length=self.maxlen)
        else:
            emb_layer = Embedding(self.embedding.shape[0], self.embedding.shape[1], input_length=self.maxlen, weights=[self.embedding], trainable=True)
        # Embedding part can try multichannel as same as origin paper
        embedding = emb_layer(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            conv = Conv1D(128, kernel_size, activation='relu')(embedding)
            convs.append(conv)
        poolings = [GlobalAveragePooling1D()(conv) for conv in convs] + [GlobalMaxPooling1D()(conv) for conv
                                                                         in convs]
        x = Concatenate()(poolings)
        x = Dropout(0.1)(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model