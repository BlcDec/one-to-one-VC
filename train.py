# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import argparse
import multiprocessing
import os

from tensorpack.callbacks.saver import ModelSaver
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated
from tensorpack.utils import logger
from tensorpack.input_source.input_source import QueueInput
from data_load import NetDataFlow
from hparam import hparam as hp
from models import Net
import tensorflow as tf


def train(args, logdir):

    # model
    model = Net()

    # dataflow
    df = NetDataFlow(hp.train.data_path, hp.train.batch_size)

    # set logger for event and model saver
    logger.set_logger_dir(logdir)

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ),)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.45  # 占用GPU90%的显存

    train_conf = TrainConfig(
        model=model,
        data=QueueInput(df(n_prefetch=1000, n_thread=4)),
        callbacks=[
            ModelSaver(checkpoint_dir=logdir),
            # TODO EvalCallback()
        ],
        max_epoch=hp.train.num_epochs,
        steps_per_epoch=hp.train.steps_per_epoch,
        # session_config=session_conf
    )
    ckpt = '{}/{}'.format(logdir, args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir)
    if ckpt:
        train_conf.session_init = SaverRestore(ckpt)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        train_conf.nr_tower = len(args.gpu.split(','))

    trainer = SyncMultiGPUTrainerReplicated(hp.train.num_gpu)

    launch_train_with_config(train_conf, trainer=trainer)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    parser.add_argument('-gpu', help='comma separated list of GPU(s) to use.')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case)
    logdir_train = '{}/train'.format(hp.logdir)

    print('case: {}, logdir: {}'.format(args.case, logdir_train))

    train(args, logdir=logdir_train)

    print("Done")