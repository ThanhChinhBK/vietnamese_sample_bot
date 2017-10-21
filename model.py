""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to build the model

See readme.md for instruction on how to run the starter code.
"""
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf



class ChatBotModel(object):
    def __init__(self, forward_only, batch_size, encode_max_size=16, decode_max_size=19,
                 hidden_size=256, num_layers=1,vocab_length=19446, num_samples=250, lr=0.01, grad_norm=5.0):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size
        self.encode_max_size = encode_max_size
        self.decode_max_size = decode_max_size
        self.hidden_size = hidden_size
        self.vocab_length = vocab_length
        self.num_samples = num_samples
        self.num_layers = num_layers
        self.lr = lr
        self.grad_norm = grad_norm

    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(self.encode_max_size)]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(self.decode_max_size + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(self.decode_max_size + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = self.decoder_inputs[1:]
        
    def _inference(self):
        print('Create inference')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if self.num_samples > 0 and self.num_samples < self.vocab_length:
            w = tf.get_variable('proj_w', [self.hidden_size, self.vocab_length])
            b = tf.get_variable('proj_b', [self.vocab_length])
            self.output_projection = (w, b)

        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, inputs, labels, 
                                              self.num_samples, self.vocab_length)
        self.softmax_loss_function = sampled_loss

        single_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)

    def _create_loss(self):
        print('Creating loss... \nIt might take a couple of minutes depending on how many buckets you have.')
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=self.vocab_length,
                    num_decoder_symbols=self.vocab_length,
                    embedding_size=self.hidden_size,
                    output_projection=self.output_projection,
                    feed_previous=do_decode)

        if self.fw_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks, 
                                        [(self.encode_max_size, self.decode_max_size)], 
                                        lambda x, y: _seq2seq_f(x, y, True),
                                        softmax_loss_function=self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                
                self.outputs[0] = [tf.matmul(output, 
                                             self.output_projection[0]) + self.output_projection[1]
                                   for output in self.outputs[0]]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks,
                                        [(self.encode_max_size, self.decode_max_size)], 
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss_function)
        print('Time:', time.time() - start)

    def _creat_optimizer(self):
        print('Create optimizer... \nIt might take a couple of minutes depending on how many buckets you have.')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                
                    
                clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[0], 
                                                                          trainables),
                                                             self.grad_norm)
                self.gradient_norms = norm
                self.train_ops = self.optimizer.apply_gradients(zip(clipped_grads, trainables), 
                                                            global_step=self.global_step)
                print('Creating opt took {} seconds'.format(time.time() - start))



    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()
