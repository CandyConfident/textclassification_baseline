from keras import Input, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense


class FastText(object):
    def __init__(self, maxlen, input_dim, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid',embedding=None):
        self.maxlen = maxlen
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

        embedding = emb_layer(input)

        x = GlobalAveragePooling1D()(embedding)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model