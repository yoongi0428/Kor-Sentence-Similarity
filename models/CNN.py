# -*- coding: utf-8 -*-
import tensorflow as tf

class Char_CNN():
    def __init__(self, config, fc_layers, filter_sizes):
        self.embedding = config.emb
        self.strmaxlen = config.strmaxlen
        self.character_size = config.charsize

        self.fc_layers = fc_layers
        self.filter_sizes = filter_sizes
        self.num_filters = config.filter_num

        self.learning_rate = config.lr
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.x1 = tf.placeholder(tf.int32, [None, self.strmaxlen])
        self.x2 = tf.placeholder(tf.int32, [None, self.strmaxlen])
        self.y_ = tf.placeholder(tf.float32, [None])

        # Embedding
        init = tf.contrib.layers.xavier_initializer(uniform=False)
        char_embedding = tf.get_variable('char_embedding', [self.character_size, self.embedding], initializer=init)

        embedded1 = tf.nn.embedding_lookup(char_embedding, self.x1)
        embedded2 = tf.nn.embedding_lookup(char_embedding, self.x2)

        self.embedded1_expanded = tf.expand_dims(embedded1, -1)
        self.embedded2_expanded = tf.expand_dims(embedded2, -1)

        pooled_outputs1 = []
        pooled_outputs2 = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv1 = tf.nn.conv2d(
                    self.embedded1_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv1")

                conv2 = tf.nn.conv2d(
                    self.embedded2_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                # Apply nonlinearity
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name="relu1")
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")
                # Maxpooling over the outputs
                pooled1 = tf.nn.max_pool(
                    h1,
                    ksize=[1, self.strmaxlen - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool1")
                pooled2 = tf.nn.max_pool(
                    h2,
                    ksize=[1, self.strmaxlen - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool2")
                pooled_outputs1.append(pooled1)
                pooled_outputs2.append(pooled2)


        fc_input_dim = self.num_filters * len(self.filter_sizes)
        self.h_pool1 = tf.concat(pooled_outputs1, 3)
        self.h_pool2 = tf.concat(pooled_outputs2, 3)

        self.fc_in1 = tf.reshape(self.h_pool1, [-1, fc_input_dim])
        self.fc_in2 = tf.reshape(self.h_pool2, [-1, fc_input_dim])

        for i, fc_output_dim in enumerate(self.fc_layers):
            with tf.name_scope("fc-layer-%s" % i):
                W = self.weight_variable([fc_input_dim, fc_output_dim])
                b = self.bias_variable([fc_output_dim])
                self.fc_out1 = tf.nn.relu(tf.nn.xw_plus_b(self.fc_in1, W, b, name="fc-out1"))
                self.fc_out2 = tf.nn.relu(tf.nn.xw_plus_b(self.fc_in2, W, b, name="fc-out2"))

            fc_input_dim = fc_output_dim
            self.fc_in1 = self.fc_out1
            self.fc_in2 = self.fc_out2

        abs_distance = tf.reduce_sum(tf.abs(self.fc_out1 - self.fc_out2), axis=1)

        # 1
        self.output_prob = tf.exp(-abs_distance)

        # # 2
        # with tf.name_scope("output-layer"):
        #     W = self.weight_variable([1, 1])
        #     b = self.bias_variable([1])
        #     self.output_prob = tf.nn.xw_plus_b(abs_distance, W, b, name="Output")


        # loss와 optimizer
        self.loss = tf.losses.mean_squared_error(self.y_ , self.output_prob)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
        self.train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
