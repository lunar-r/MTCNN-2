# -*- coding: utf-8 -*-

from keras.optimizers import Adam, SGD
import os

from model_net import Pnet, Rnet, Onet
from load_data import DataGenerator

filepath = os.path.dirname(os.path.abspath(__file__))

GAN_DATA_ROOT_DIR = filepath + '/data/traindata/'

net_name = 'Rnet'

model_file = filepath + '/model/%s.h5' %(net_name)


def train_with_data_generator(dataset_root_dir=GAN_DATA_ROOT_DIR, model_file=model_file, weights_file=None):

    batch_size = 32*7
    epochs = 30
    learning_rate = 0.001
    
    pos_dataset_path = os.path.join(dataset_root_dir, 'pos_shuffle_%s.h5' %(net_name))
    neg_dataset_path = os.path.join(dataset_root_dir, 'neg_shuffle_%s.h5' %(net_name))
    part_dataset_path = os.path.join(dataset_root_dir, 'part_shuffle_%s.h5' %(net_name))
    landmarks_dataset_path = os.path.join(dataset_root_dir, 'landmarks_shuffle_%s.h5' %(net_name))

    data_generator = DataGenerator(pos_dataset_path, neg_dataset_path, part_dataset_path, landmarks_dataset_path, batch_size, im_size=12)
    data_gen = data_generator.generate()
    steps_per_epoch = data_generator.steps_per_epoch()

    if net_name == 'Pnet':
        _net = Pnet()
    elif net_name == 'Rnet':
        _net = Rnet()
    else:
        _net = Onet()

    _net_model = _net.model(training=True)
    _net_model.summary()
    if weights_file is not None:
        _net_model.load_weights(weights_file)

    #sgd = SGD(lr=0.005, momentum=0.8)
    #_p_net_model.compile(optimizer=sgd, loss=_p_net.loss_func, metrics=[_p_net.accuracy, _p_net.recall])
    _net_model.compile(Adam(lr=learning_rate), loss=_net.loss_func, metrics=[_net.accuracy, _net.recall])

    _net_model.fit_generator(data_gen,
                                steps_per_epoch=steps_per_epoch,
                                initial_epoch=0,
                                epochs=epochs)

    _net_model.save_weights(model_file)

if __name__ == '__main__':

    train_with_data_generator()