from keras import Input, Model
from keras.layers import Embedding, Dense, Dropout, CuDNNLSTM as LSTM, Bidirectional,Lambda


class TextRNN(object):
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
            emb_layer = Embedding(self.input_dim, self.embedding_dims,
                                  input_length=self.maxlen)
        else:
            emb_layer = Embedding(self.embedding.shape[0], self.embedding.shape[1],
                                  input_length=self.maxlen, weights=[self.embedding], trainable=True)
        embedding = emb_layer(input)
        #return_sequences 这个参数如果直接接全连接层，就可以设置成False，如果还有别的操作，设置成True
        x = Bidirectional(LSTM(128, return_sequences=True), merge_mode="sum")(embedding)
        x = Lambda(lambda x: x[:, 0, :], name="extract_layer")(x)
        x = Dropout(0.5)(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model