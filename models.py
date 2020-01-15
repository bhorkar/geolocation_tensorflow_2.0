
from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.python.feature_column import feature_column_v2 as fc
from preprocess import normalize, create_stats, _pre_process
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.framework.sparse_tensor import is_sparse
from tensorflow.python.framework import ops
from tensorflow.python.framework.sparse_tensor import SparseTensor
from utils.config_data import unit
from config import window_size
def make_input_layers(dataset, feature_columns, batch_size):
    features = dataset[0]
    labels = dataset[1]
    for key, tensor in features.items():
        if key in ['rsrp', 'dt', 'rsrp0', 'rsrp1', 'rsrp2', 'ta']:
            if isinstance(tensor, ops.Tensor):
                if not isinstance(tensor, SparseTensor):
                    features[key] = tf.contrib.layers.dense_to_sparse(
                        tensor, eos_token=-2)

    sequence_input_layer = SequenceFeatures(
        [
            feature_columns['gci'],
            feature_columns['ta'],
            feature_columns['rsrp'],
            feature_columns['gci0'],
            feature_columns['gci1'],
            feature_columns['gci2'],
            feature_columns['rsrp0'],
            feature_columns['rsrp1'],
            feature_columns['rsrp2'],
            feature_columns['dt']])
    sequence_input, sequence_length = sequence_input_layer(features)
    sequence_length_mask = tf.sequence_mask(sequence_length)
    return sequence_input, sequence_length_mask


def get_train_op(loss, params):
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        params['learning_rate'],
        global_step,
        params['decay_steps'],
        params['decay_rate'],
        staircase=False
    )
    optimizer = tf.train.AdamOptimizer()
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss, global_step=global_step)


def _model_fn(features, labels, mode, params):
    """A model_fn that builds the DNN regression spec.
    Args:
      mode (tf.estimator.ModeKeys): One of ModeKeys.(TRAIN|PREDICT|INFER) which
        is used to selectively add operations to the graph.
      features (Mapping[str:Tensor]): Input features for the model.
      labels (Tensor): Label Tensor.
    Returns:
      tf.estimator.EstimatorSpec which defines the model. Will have different
      populated members depending on `mode`. See:
        https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
      for details.
    """

    with tf.variable_scope('geolocation_model', reuse=tf.AUTO_REUSE):

        sequence_input, sequence_length_mask = make_input_layers(
            [features, labels],
            params['feature_columns'], params['batch_size']
        )
        if mode != tf.estimator.ModeKeys.PREDICT:
            label_words = tf.contrib.lookup.index_table_from_file(os.path.join(
                params['transform_dir'], 'tftransform_tmp/label'), num_oov_buckets=0)
            tags = label_words.lookup(labels['LAT_LON_10'])

        table_index_to_string = tf.contrib.lookup.index_to_string_table_from_file(
            os.path.join(params['transform_dir'], 'tftransform_tmp/label'),
            default_value='0',
        )
        layer_1 = tf.keras.layers.LSTM(
            params['hidden_units'][0],
            return_sequences=True)
        LSTM_output = layer_1(sequence_input, mask=sequence_length_mask)
        for l in range(1, params['nlayers_stack'] - 2):
            LSTM_output = tf.keras.layers.LSTM(
                params['hidden_units'][l],
                return_sequences=True)(LSTM_output)
        layer_n = tf.keras.layers.LSTM(
            params['hidden_units'][-1], return_sequences=False)(LSTM_output)
        d = tf.keras.layers.Dense(params['hidden_units'][-1] / 2)(layer_n)
        d = tf.keras.layers.Activation('relu')(d)  # 32 not better than 16
        d = tf.keras.layers.BatchNormalization()(d)
        #    d, training=mode == tf.estimator.ModeKeys.TRAIN)  # 32 not better than 16
        d = tf.keras.layers.Dense(params['label_size'])(d)
        left = tf.keras.layers.Activation('softmax')(d)  # geo matrix
        utm = tf.keras.layers.Dense(2)(left)

        predictions_ll = utm
        if mode == tf.estimator.ModeKeys.PREDICT:
            predicted_label = table_index_to_string.lookup(
                tf.argmax(left, axis=1))
            sparse_lat_lon = tf.string_split(predicted_label, delimiter='_')
            nrow = predicted_label.get_shape().as_list()

            dense_lat_lon = tf.sparse_to_dense(
                sparse_lat_lon.indices,
                sparse_lat_lon.dense_shape,
                sparse_lat_lon.values,
                default_value='0'
            )
            basic = tf.stack([utm[:,0], utm[:,1]], axis=1)
            lat = tf.strings.to_number(dense_lat_lon[:, 0])
            lon = tf.strings.to_number(dense_lat_lon[:, 1])
            lat_grid = lat * unit + unit / 2
            lon_grid = lon * unit + unit / 2
            lat_regression = utm[:, 0] * \
                params['lat_stat'][1] + params['lat_stat'][0]
            lon_regression = utm[:, 1] * \
                params['lon_stat'][1] + params['lon_stat'][0]
            lat = (lat_grid + lat_regression) / 2
            lon = (lon_grid + lon_regression) / 2
            predictions = tf.stack([lat, lon], axis=1)
            predictions_grid = tf.stack([lat_grid, lon_grid], axis=1)
            predictions_regression = tf.stack(
                [lat_regression, lon_regression], axis=1)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "lat/lon": predictions,
                    "grid_pred": predictions_grid,
                    "regression_pred": predictions_regression,
                    "basic":  basic,
                    "features": features['gci0']
                })

        total_loss_ll = tf.losses.absolute_difference(
            labels['lat'], predictions_ll[:, 0]) + tf.losses.absolute_difference(labels['lon'], predictions_ll[:, 1])
        cross_entrophy_loss = tf.losses.sparse_softmax_cross_entropy(
            tags, d)
        total_loss = params['weight_regression'] * total_loss_ll + \
            params['weight_cross_entrophy'] * cross_entrophy_loss
        #average_loss = total_loss / tf.to_float(params['batch_size'])
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = get_train_op(total_loss, params['train'])

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)
# In evaluation mode we will calculate evaluation metrics.
        assert mode == tf.estimator.ModeKeys.EVAL
        mae = tf.metrics.mean_absolute_error(tf.stack([labels['lat'], labels['lon']], axis = 1), predictions_ll[:,:]) 
        #mae = (tf.metrics.mean_absolute_error(
         #   labels['lat'], predictions_ll[:, 0]) + tf.metrics.mean_absolute_error(labels['lon'], predictions_ll[:, 1]))
        accuracy = tf.metrics.accuracy(
            labels=tags, predictions=(
                tf.argmax(
                    left, axis=1)))
  # Add the rmse to the collection of evaluation metrics.
        eval_metrics = {"mae": mae}
        tf.summary.scalar('train_accuracy', accuracy[1])
        tf.summary.scalar('mae', mae)

        eval_metrics = {"mae": mae, "accuracy": accuracy}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metrics)


def serving_input_receiver_fn():
    inputs = {}
    for feature_name in ['ta', 'rsrp', 'rsrp0', 'rsrp1', 'rsrp2', 'dt']:
        inputs[feature_name] = tf.placeholder(
            tf.float32, [None, window_size], name=feature_name)
    for feature_name in ['gci', 'gci0', 'gci1', 'gci2']:
        inputs[feature_name] = tf.placeholder(
            tf.string, [None, window_size], name=feature_name)
    features = {}
    for key, tensor in inputs.items():
        features[key] = tensor

    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=inputs
    )


def save_model_raw_features(
        estimator,
        transform_dir,
        serving_output_dir,
        filepath):
 # estimator = tf.contrib.estimator.forward_features(estimator, keys="row_identifier")
 #   tf.gfile.DeleteRecursively(serving_output_dir)

    export_dir = estimator.export_saved_model(
        export_dir_base=serving_output_dir,
        serving_input_receiver_fn=serving_input_receiver_fn)

    # test_export_model_raw(export_dir);

    print("saving in", export_dir)
    return export_dir


def save_model_raw_features_csv(
        model,
        transform_dir,
        serving_output_dir,
        filepath):
    """Returns serving_input_receiver_fn for csv.

  The input arguments are relevant to `tf.decode_csv()`.

  Args:
    column_names: a list of column names in the order within input csv.
    column_defaults: a list of default values with the same size of
        column_names. Each entity must be either a list of one scalar, or an
        empty list to denote the corresponding column is required.
        e.g. [[""], [2.5], []] indicates the third column is required while
            the first column must be string and the second must be float/double.

  Returns:
    a serving_input_receiver_fn that handles csv for serving.
    """
    def serving_input_receiver_fn():
        csv = tf.placeholder(dtype=tf.string, shape=[None], name="csv")
        features, labels = _pre_process(csv)
        receiver_tensors = {"inputs": csv}
        return tf.estimator.export.ServingInputReceiver(
            features, receiver_tensors)

    export_dir = model.export_saved_model(serving_output_dir,
                                          serving_input_receiver_fn)
    print("saving in", export_dir)
    return export_dir


def save_model(model, transform_dir, serving_output_dir, filepath):
    # code needs to be modified
    # this code uses contrib which is deprecated.
    # However, there is an issue
    # https://github.com/tensorflow/tensorflow/issues/30958

    from tensorflow.python.feature_column import feature_column_lib as fc
    from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as seq_fc
    vocab_file = os.path.join(transform_dir, 'tftransform_tmp/gci')
    stats_dict = create_stats(filepath)

    def make_columns():
        """
        Builds the feature_columns required by the estimator to link the Dataset and the model_fn
        :return:
        """
        columns_dict = {}

        columns_dict['gci'] = fc.indicator_column(
            fc.sequence_categorical_column_with_vocabulary_file(
                'gci',
                vocab_file,
                default_value="0"
            )
        )
        columns_dict['ta'] = (
            seq_fc.sequence_numeric_column(
                'ta', normalizer_fn=lambda x: normalize(x, 'ta', stats_dict)
            )
        )
        columns_dict['rsrp'] = (
            seq_fc.sequence_numeric_column(
                'rsrp', normalizer_fn=lambda x: normalize(
                    x, 'rsrp', stats_dict)))
        columns_dict['gci0'] = fc.indicator_column(
            fc.sequence_categorical_column_with_vocabulary_file(
                'gci0',
                vocab_file,
                default_value="0"
            )
        )
        columns_dict['rsrp0'] = (
            seq_fc.sequence_numeric_column(
                'rsrp0', normalizer_fn=lambda x: normalize(
                    x, 'rsrp0', stats_dict)))
        columns_dict['gci1'] = fc.indicator_column(
            fc.sequence_categorical_column_with_vocabulary_file(
                'gci1',
                vocab_file,
                default_value="0"
            )
        )
        columns_dict['rsrp1'] = (
            seq_fc.sequence_numeric_column(
                'rsrp1', normalizer_fn=lambda x: normalize(
                    x, 'rsrp1', stats_dict)))
        columns_dict['gci2'] = fc.indicator_column(
            fc.sequence_categorical_column_with_vocabulary_file(
                'gci2',
                vocab_file,
                default_value="0"
            )
        )
        columns_dict['rsrp2'] = (
            seq_fc.sequence_numeric_column(
                'rsrp2', normalizer_fn=lambda x: normalize(
                    x, 'rsrp2', stats_dict)))
        columns_dict['dt'] = (
            seq_fc.sequence_numeric_column(
                'dt', normalizer_fn=lambda x: normalize(x, 'dt', stats_dict)
            )
        )
        return columns_dict
    f = make_columns()
    feature_columns = [
        f['gci'], f['ta'], f['rsrp'],
        f['gci0'], f['rsrp0'],
        f['gci1'], f['rsrp1'],
        f['gci2'], f['rsrp2'],
        f['dt']
    ]
    feature_spec = fc.make_parse_example_spec(feature_columns)
    print("**********************feature_spec************************************")
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = model.export_savedmodel(serving_output_dir,
                                         serving_input_receiver_fn)
    print("saving in", export_dir)
    return export_dir
