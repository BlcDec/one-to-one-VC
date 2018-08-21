# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils import (
    get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack_extension
from hparam import hparam as hp
from modules import prenet, cbhg, normalize

class Net(ModelDesc):

    def _get_inputs(self):
        n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1  #401

        return [InputDesc(tf.float32, (None, n_timesteps, hp.default.n_mels), 'r_mel'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_fft // 2 + 1), 't_spec'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_mels), 't_mel'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_fft // 2 + 1), 'r_spec')]

    def _build_graph(self, inputs):
        self.r_mel, self.t_spec, self.t_mel, self.r_spec = inputs

        is_training = get_current_tower_context().is_training

        # build net
        with tf.variable_scope('net'):
            self.pred_spec, self.pred_mel = self.network(self.r_mel, is_training)
        self.pred_spec = tf.identity(self.pred_spec, name='pred_spec')

        self.cost = self.loss()

        # summaries
        tf.summary.scalar('net/train/loss', self.cost)

        if not is_training:
            tf.summary.scalar('net/eval/summ_loss', self.cost)

    def _get_optimizer(self):
        gradprocs = [
            tensorpack_extension.FilterGradientVariables('.*net.*', verbose=False),
            gradproc.MapGradient(
                lambda grad: tf.clip_by_value(grad, hp.train.clip_value_min, hp.train.clip_value_max)),
            gradproc.GlobalNormClip(hp.train.clip_norm),
            # gradproc.PrintGradient(),
            # gradproc.CheckGradient(),
        ]
        lr = tf.get_variable('learning_rate', initializer=hp.train.lr, trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        return optimizer.apply_grad_processors(opt, gradprocs)

    @auto_reuse_variable_scope
    def network(self, mels, is_training):
        # Pre-net
        prenet_out = prenet(mels,
                            num_units=[hp.train.hidden_units, hp.train.hidden_units // 2],
                            dropout_rate=hp.train.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # CBHG1: mel-scale
        pred_mel = cbhg(prenet_out, hp.train.num_banks, hp.train.hidden_units // 2,
                        hp.train.num_highway_blocks, hp.train.norm_type, is_training,
                        scope="cbhg_mel")
        pred_mel = tf.layers.dense(pred_mel, self.t_mel.shape[-1], name='pred_mel')  # (N, T, n_mels)

        # CBHG2: linear-scale
        pred_spec = tf.layers.dense(pred_mel, hp.train.hidden_units // 2)  # (N, T, n_mels)
        pred_spec = cbhg(pred_spec, hp.train.num_banks, hp.train.hidden_units // 2,
                   hp.train.num_highway_blocks, hp.train.norm_type, is_training, scope="cbhg_linear")
        pred_spec = tf.layers.dense(pred_spec, self.t_spec.shape[-1], name='pred_spec')  # log magnitude: (N, T, 1+n_fft//2)

        return pred_spec, pred_mel

    def loss(self):
        loss_spec = tf.reduce_mean(tf.squared_difference(self.pred_spec, self.t_spec))
        loss_mel = tf.reduce_mean(tf.squared_difference(self.pred_mel, self.t_mel))
        loss = loss_spec + loss_mel
        return loss