from absl import app as absl_app
from absl import flags, logging
import tensorflow as tf
import os
import json
import socket
ip_address = socket.gethostbyname(socket.gethostname())
session_config = tf.ConfigProto()
session_config.gpu_options.per_process_gpu_memory_fraction = 0.1
session_config.gpu_options.allow_growth = True
session = tf.Session(config=session_config)
sess = tf.Session()



sess = tf.Session()

hparams_dict = {}
hparams_dict['0'] = {'weight_regression':0.25, 'weight_cross_entrophy':0.05, 'nlayers_stack':2, 'hidden_units':32}
#hparams_dict['1'] = {'weight_regression':0.5, 'weight_cross_entrophy':0.5, 'nlayers_stack':2, 'hidden_units':64}
#hparams_dict['2'] = {'weight_regression':0.25, 'weight_cross_entrophy':0.05, 'nlayers_stack':2, 'hidden_units':128}
#hparams_dict['3'] = {'weight_regression':0.25, 'weight_cross_entrophy':0.05, 'nlayers_stack':4, 'hidden_units':64}


strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
window_size = 2
