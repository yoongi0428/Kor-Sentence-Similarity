# -*- coding: utf-8 -*-
import tensorflow as tf

class MLP():
    def __init__(self, config, fc_hidden):
        self.embedding = config.emb
        self.strmaxlen = config.strmaxlen
        self.input_size = self.embedding * self.strmaxlen

        self.learning_rate = config.lr
        self.character_size = config.charsize

        self.x1 = tf.placeholder(tf.int32, [None, self.strmaxlen])
        self.x2 = tf.placeholder(tf.int32, [None, self.strmaxlen])
        self.y_ = tf.placeholder(tf.float32, [None])

        # Embedding
        init = tf.contrib.layers.xavier_initializer(uniform=False)
        char_embedding = tf.get_variable('char_embedding', [self.character_size, self.embedding], initializer=init)

        out1 = tf.nn.embedding_lookup(char_embedding, self.x1)
        out2 = tf.nn.embedding_lookup(char_embedding, self.x2)

        out1 = tf.reshape(out1, [-1, self.input_size])
        out2 = tf.reshape(out2, [-1, self.input_size])

        dim = self.input_size
        for i, fc_dim in enumerate(fc_hidden):
            with tf.name_scope("Layer-" + str(i)):
                weight = self.weight_variable([dim, fc_dim])
                bias = self.bias_variable([fc_dim])

                out1 = tf.nn.tanh(tf.nn.xw_plus_b(out1, weight, bias, name='out1-' + str(i)))
                out2 = tf.nn.tanh(tf.nn.xw_plus_b(out2, weight, bias, name='out2-' + str(i)))

            dim = fc_dim

        # Similarity = exp( -sum( Manhattan Distance ) )
        dist = tf.reduce_sum(tf.abs(out1 - out2), axis=1)
        self.output_prob = tf.exp(-dist)

        # loss & optimizer
        self.loss = tf.losses.mean_squared_error(self.y_ , self.output_prob)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients = [
            None if g is None else tf.clip_by_norm(g, 1.0)
            for g in gradients
        ]
        self.train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)