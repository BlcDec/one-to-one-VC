# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from models import Net
import argparse
from hparam import hparam as hp
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
from data_load import NetDataFlow


def get_eval_input_names():
    return ['r_mel', 't_spec']


def get_eval_output_names():
    return ['net/eval/summ_loss']


def eval(logdir):
    # Load graph
    model = Net()

    # dataflow
    df = NetDataFlow(hp.test.data_path, hp.test.batch_size)

    ckpt = tf.train.latest_checkpoint(logdir)
    session_inits = []
    if ckpt:
        session_inits.append(SaverRestore(ckpt))
    pred_conf = PredictConfig(
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names(),
        session_init=ChainInit(session_inits))
    predictor = OfflinePredictor(pred_conf)

    r_mel, t_spec, _ = next(df().get_data())
    summ_loss, = predictor(r_mel, t_spec)

    writer = tf.summary.FileWriter(logdir)
    writer.add_summary(summ_loss)
    writer.close()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name of train')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case)
    logdir_train = '{}/{}/train'.format(hp.logdir_path, args.case)

    eval(logdir=logdir_train)

    print("Done")
