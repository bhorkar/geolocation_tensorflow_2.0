from __future__ import print_function

#from tensorflow.python import debug as tf_debug
import os
import shutil
from datetime import datetime
from absl import app as absl_app
from absl import flags, logging
import tensorflow as tf
from utils.dataformat import convert_train_files_format
from preprocess import csv_input_fn, compute_gci_vocab, make_columns, create_stats
from models import _model_fn, save_model_raw_features, serving_input_receiver_fn
from utils.prediction_array import compute_error,compute_error_from_pred
from config import *
import six
tf.set_random_seed(1234)
print("GPU Available: ", tf.test.is_gpu_available())


flags.DEFINE_string(
    'filename', '3_62263-.csv',
    'filename for enb'
)
flags.DEFINE_integer('preprocess', 1, 'preprocess data')
flags.DEFINE_integer('epoch', 5, 'preprocess data')
flags.DEFINE_integer('batch_size', 1024, 'batch_size')
flags.DEFINE_integer('train', 1, 'perform training')
flags.DEFINE_string('model_output', './output/model/', 'preprocess data')
flags.DEFINE_string(
    'schema_path',
    './data/schema_input_data.pbtxt',
    'schema path')
flags.DEFINE_string('train_raw_data', './raw_data/',
                    'Directory where training data is present.')
flags.DEFINE_string(
    'transform_dir',
    './tft_train_data/',
    'store the transformed data')
flags.DEFINE_string('serving_output_dir', './model/', 'serving output dir')
flags.DEFINE_string(
    'prediction_dir',
    './prediction_output',
    'prediction output dir')
flags.DEFINE_string('processed_data', './data/',
                    'Directory where training data is present.')
FLAGS = flags.FLAGS


def model_main(FLAGS):
    if FLAGS.preprocess == 1:
        shutil.rmtree(FLAGS.transform_dir, ignore_errors=True)
        shutil.rmtree(FLAGS.model_output, ignore_errors=True)
        shutil.rmtree(FLAGS.serving_output_dir, ignore_errors=True)
        shutil.rmtree(FLAGS.transform_dir, ignore_errors=True)
        tf.io.gfile.makedirs(FLAGS.transform_dir)
        tf.io.gfile.makedirs(FLAGS.model_output)
        tf.io.gfile.makedirs(FLAGS.serving_output_dir)
        tf.io.gfile.makedirs(FLAGS.transform_dir)
        convert_train_files_format(
            FLAGS.train_raw_data,
            FLAGS.processed_data,
            FLAGS.filename)
        compute_gci_vocab(
            input_handle=os.path.join(
                FLAGS.train_raw_data,
                FLAGS.filename),
            working_dir=FLAGS.transform_dir,
            schema_file=FLAGS.schema_path,
            pipeline_args=['--runner=DirectRunner'])
    if FLAGS.train == 1:
        hparams = {
            'batch_size': FLAGS.batch_size,
            'num_epochs': FLAGS.epoch,
            'transform_dir': FLAGS.transform_dir,
            'train': {
                'learning_rate': 0.001,
                'decay_rate': 0.98,
                'decay_steps': 2000
            },
        }
        with sess.as_default():
            label_words = tf.contrib.lookup.index_table_from_file(os.path.join(
                FLAGS.transform_dir, 'tftransform_tmp/label'), num_oov_buckets=0)
            sess.run(tf.tables_initializer())
            label_size = ((label_words.size().eval()))
        hparams['label_size'] = label_size
        train_filepath = os.path.join(
            os.path.join(
                FLAGS.processed_data,
                'train'),
            FLAGS.filename)
        hparams['feature_columns'], stats = make_columns(
            FLAGS.transform_dir, train_filepath)
        hparams['lat_stat'] = stats['lat']
        hparams['lon_stat'] = stats['lon']
        eval_filepath = os.path.join(
            os.path.join(
                FLAGS.processed_data,
                'eval'),
            FLAGS.filename)
        test_filepath = os.path.join(
            os.path.join(
                FLAGS.processed_data,
                'test'),
            FLAGS.filename)
        logging.info('started training')
        print("starting training")
        min_metric_per_hparam_value = float('inf')
        best_params = None
        for k, param_values in six.iteritems(hparams_dict):
            print(param_values)
            for param, val in six.iteritems(param_values):
                if param == 'hidden_units':
                    val = [val] * param_values['nlayers_stack']
                hparams[param] = val

               # model_dir_str += str(val)
            print(hparams)
            append_model_str = '_'.join(str(x) for x in param_values.values())
            model_output_dir = os.path.join(
                FLAGS.model_output, append_model_str)
            serving_output_dir = os.path.join(
                FLAGS.serving_output_dir, append_model_str)
            run_config = tf.estimator.RunConfig(
                model_dir=model_output_dir,
                save_checkpoints_steps=500,
                keep_checkpoint_max=1,
            )
            estimator = tf.estimator.Estimator(
                model_fn=_model_fn,
                config=run_config,
                params=hparams
            )
            early_stopping_loss = tf.estimator.experimental.stop_if_no_decrease_hook(
                estimator, metric_name='loss', max_steps_without_decrease=2000, min_steps=100)
            early_stopping_mae = tf.estimator.experimental.stop_if_no_decrease_hook(
                estimator, metric_name='mae', max_steps_without_decrease=2000, min_steps=100)

            early_stopping_accuracy = tf.estimator.experimental.stop_if_no_increase_hook(
                estimator, metric_name='accuracy', max_steps_without_increase=2000, min_steps=100)

            train_spec = tf.estimator.TrainSpec(
                input_fn=lambda: csv_input_fn(
                    train_filepath,
                    hparams['batch_size'],
                    hparams['num_epochs']),
                hooks=[early_stopping_loss])
            exporter = tf.estimator.BestExporter(
                name="best_exporter",
                serving_input_receiver_fn=serving_input_receiver_fn,
                exports_to_keep=5)

            eval_spec = tf.estimator.EvalSpec(
                input_fn=lambda: csv_input_fn(
                    eval_filepath,
                    hparams['batch_size'],
                    1),
                exporters=exporter)
            time_start = datetime.utcnow()
            print("Experiment started at {}".format(
                time_start.strftime("%H:%M:%S")))
            print(".......................................")

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
            pred = estimator.predict(
                    input_fn=lambda: csv_input_fn(
                    test_filepath,
                    hparams['batch_size'],
                    1)
            )
            save_model_raw_features(
                estimator,
                FLAGS.transform_dir,
                serving_output_dir,
                train_filepath)
            os.system("mkdir -p {}".format(FLAGS.prediction_dir))
            median = compute_error(
                serving_output_dir,
                FLAGS.filename,
                FLAGS.transform_dir,
                datapath=os.path.join(FLAGS.processed_data, 'eval'),
                append_str=append_model_str,
                prediction_dir = FLAGS.prediction_dir)
            median = compute_error_from_pred(
                pred,
                serving_output_dir,
                FLAGS.filename,
                FLAGS.transform_dir,
                datapath=os.path.join(FLAGS.processed_data, 'eval'),
                append_str=append_model_str,
                prediction_dir = FLAGS.prediction_dir
            )
            if min_metric_per_hparam_value > median:
                min_metric_per_hparam_value = median
                best_params = param_values
        time_end = datetime.utcnow()
        median = compute_error_from_pred(
                pred,
                serving_output_dir,
                FLAGS.filename,
                FLAGS.transform_dir,
                datapath=os.path.join(FLAGS.processed_data, 'test'),
                append_str=append_model_str,
                prediction_dir = FLAGS.prediction_dir
            )
        print("best parameters are", best_params)
        print("Median accuracy achieved", min_metric_per_hparam_value)
        print("copying best model")
        os.system("cp -r {} {}".format(serving_output_dir,
                                       os.path.join(FLAGS.serving_output_dir, 'best_model')))
        print(".......................................")
        print(
            "Experiment finished at {}".format(
                time_end.strftime("%H:%M:%S")))

        # tf.gfile.MkDir(FLAGS.prediction_dir)
        time_elapsed = time_end - time_start
        print(
            "Experiment elapsed time: {} seconds".format(
                time_elapsed.total_seconds()))


def main(_):
    model_main(flags.FLAGS)
# ./tutorials/rnn/quickdraw/train_model.py


if __name__ == "__main__":
    # define flags
    absl_app.run(main)
