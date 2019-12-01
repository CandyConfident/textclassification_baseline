# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Bidirectional, CuDNNGRU, TimeDistributed

from attention import Attention


class HAN(object):
    def __init__(self, max_sents, max_sen_len, input_dim, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid',embedding=None):
        self.max_sents = max_sents
        self.max_sen_len = max_sen_len
        # Size of the vocabulary,
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = embedding

    def get_model(self):
        # Word part
        sentence_input = Input(shape=(self.max_sen_len,),dtype='int32')
        if not self.embedding:
            emb_layer = Embedding(self.input_dim, self.embedding_dims,
                                  input_length=self.max_sen_len,trainable=True, mask_zero=True)
        else:
            emb_layer = Embedding(self.embedding.shape[0], self.embedding.shape[1],
                                  input_length=self.max_sen_len, weights=[self.embedding],
                                  trainable=True,mask_zero=True)

        embedded_sequences = emb_layer(sentence_input)
        gru_word = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedded_sequences)
        word_atten = Attention(128)(gru_word)
        model_word = Model(sentence_input, word_atten)

        # Sentence part
        input = Input(shape=(self.max_sents, self.max_sen_len))
        x_sentences = TimeDistributed(model_word)(input)
        gru_sent = Bidirectional(CuDNNGRU(128, return_sequences=True))(x_sentences)
        sent_atten = Attention(128)(gru_sent)
        output = Dense(self.class_num, activation=self.last_activation)(sent_atten)
        model = Model(inputs=input, outputs=output)
        return model