import os
import json
import shutil
import csv
import tensorflow as tf
import apache_beam as beam
from google.protobuf import text_format
import tensorflow_data_validation as tfdv
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
import tensorflow_transform.beam as tft_beam
from tensorflow.python.lib.io import file_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from google.protobuf.json_format import MessageToJson
from utils.config_data import setcolumn_list_original
import tensorflow_transform as transform
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow_data_validation.utils import schema_util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from config import window_size
_CSV_COLUMNS_NAMES, _CSV_COLUMN_DEFAULTS, _CSV_COLUMN_types, _UNUSED = setcolumn_list_original()
stats_dict = None


def add_tensor(sp_x, y):
    y = ops.convert_to_tensor(y, dtype=sp_x.dtype.base_dtype, name="y")
    return sparse_tensor.SparseTensor(
        sp_x.indices,
        gen_sparse_ops.sparse_dense_cwise_add(
            sp_x.indices,
            sp_x.values,
            sp_x.dense_shape,
            y,
            name="add"),
        sp_x.dense_shape)


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '0' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def create_stats(transform_dir, filepath):
    DENSE_FLOAT_FEATURE_KEYS = []
    VOCAB_FEATURE_KEYS = []
    for i in range(len(_CSV_COLUMNS_NAMES)):
        if _CSV_COLUMN_types[i] is tf.string:
            VOCAB_FEATURE_KEYS.append(_CSV_COLUMNS_NAMES[i])
        if _CSV_COLUMN_types[i] is tf.float32:
            DENSE_FLOAT_FEATURE_KEYS.append(_CSV_COLUMNS_NAMES[i])
    train_stats = tfdv.generate_statistics_from_csv(
        data_location=filepath, delimiter='|')
    jsonObj = python_obj = json.loads(MessageToJson(train_stats))
    stats_dict = {}
    for i in range(len(jsonObj['datasets'][0]['features'])):
        mean, std, min, max = 0.1, 0.1, 0.1, 0.2
        try:
            name = jsonObj['datasets'][0]['features'][i]['path']['step'][0]
            mean = (jsonObj['datasets'][0]['features'][i]['numStats']['mean'])
            std = (jsonObj['datasets'][0]['features'][i]['numStats']['stdDev'])
            minv = (jsonObj['datasets'][0]['features'][i]['numStats']['min'])
            maxv = (jsonObj['datasets'][0]['features'][i]['numStats']['max'])
            stats_dict[name] = [
                float(mean),
                float(std),
                float(minv),
                float(maxv)]
        except BaseException as e:
            try: 
                name = jsonObj['datasets'][0]['features'][i]['name']
                mean = (jsonObj['datasets'][0]['features'][i]['numStats']['mean'])
                std = (jsonObj['datasets'][0]['features'][i]['numStats']['stdDev'])
                minv = (jsonObj['datasets'][0]['features'][i]['numStats']['min'])
                maxv = (jsonObj['datasets'][0]['features'][i]['numStats']['max'])
                stats_dict[name] = [
                    float(mean),
                    float(std),
                    float(minv),
                    float(maxv)]
            except BaseException as e1:
                pass
    # save
    for c in ['rsrp0', 'rsrp1', 'rsrp2', 'rsrp', 'ta']:
        count = 0
        try:
            statsc = stats_dict[c]
            count += 1
        except BaseException:
            statsc = [0, 0, 0, 0]
        mean, std, minv, maxv = statsc
        for i in range(1, window_size, 1):
            try:
                minv = (stats_dict[c + '_' + str(i)][2]) + minv
                maxv = (stats_dict[c + '_' + str(i)][3]) + maxv
                mean = (stats_dict[c + '_' + str(i)][0]) + mean
                std = (stats_dict[c + '_' + str(i)][1]) + std
                count += 1
            except Exception as e:
                print(e)
                pass
        if count == 0:
            stats_dict[c] = [0, 0, 0, 0]
        else:
            stats_dict[c] = [
                mean / count,
                std / count,
                minv / count,
                maxv / count]

    w = csv.writer(
        open(
            os.path.join(
                transform_dir,
                "output_stats_{}".format(
                    os.path.basename(filepath))),
            "w"))
    for key, val in stats_dict.items():
        w.writerow([key, val])

    return stats_dict


def get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def make_csv_coder(schema):
    """Return a coder for tf.transform to read csv files."""
    raw_feature_spec = get_raw_feature_spec(schema)
    parsing_schema = dataset_schema.from_feature_spec(raw_feature_spec)
    return tft_coders.CsvCoder(
        _CSV_COLUMNS_NAMES,
        parsing_schema,
        delimiter='|')


def read_schema(path):
    """Reads a schema from the provided location.
    Args:
      path: The location of the file holding a serialized Schema proto.
    Returns:
      An instance of Schema or None if the input argument is None
    """
    result = schema_pb2.Schema()
    contents = file_io.read_file_to_string(path)
    text_format.Parse(contents, result)
    return result


def compute_gci_vocab(input_handle,
                      working_dir,
                      schema_file,
                      transform_dir=None,
                      max_rows=None,
                      pipeline_args=None):
    """The main tf.transform method which analyzes and transforms data.
    Args:
      input_handle: BigQuery table name to process specified as DATASET.TABLE or
        path to csv file with input data.
      outfile_prefix: Filename prefix for emitted transformed examples
      working_dir: Directory in which transformed examples and transform function
        will be emitted.
      schema_file: An file path that contains a text-serialized TensorFlow
        metadata schema of the input data.
      transform_dir: Directory in which the transform output is located. If
        provided, this will load the transform_fn from disk instead of computing
        it over the data. Hint: this is useful for transforming eval data.
      max_rows: Number of rows to query from BigQuery
      pipeline_args: additional DataflowRunner or DirectRunner args passed to the
        beam pipeline.
    """

    def preprocessing_fn(inputs):
        """tf.transform's callback function for preprocessing inputs.
        Args:
          inputs: map from feature keys to raw not-yet-transformed features.
        Returns:
          Map from string feature key to transformed feature operations.
        """
        outputs = {}
        DENSE_FLOAT_FEATURE_KEYS = []
        VOCAB_FEATURE_KEYS = []
        _CSV_COLUMNS_NAMES, _CSV_COLUMN_DEFAULTS, _CSV_COLUMN_types, _UNUSED = setcolumn_list_original()
        for i in range(len(_CSV_COLUMNS_NAMES)):
            if _CSV_COLUMN_types[i] is tf.string:
                VOCAB_FEATURE_KEYS.append(_CSV_COLUMNS_NAMES[i])

        outputs['gci'] = tf.expand_dims(_fill_in_missing(inputs['gci']), 1)
        for key in VOCAB_FEATURE_KEYS:
            if key in _UNUSED:
                continue
            if 'gci' in key:
                appendlist = tf.expand_dims(_fill_in_missing(inputs[key]), 1)
                outputs['gci'] = tf.concat([appendlist, outputs['gci']], 0)
        transform.vocabulary(outputs['gci'], vocab_filename='gci')
        transform.vocabulary(inputs['LAT_LON_10'], vocab_filename='label')
        return outputs

    schema = read_schema(schema_file)
    raw_feature_spec = get_raw_feature_spec(schema)
    raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
    raw_data_metadata = dataset_metadata.DatasetMetadata(raw_schema)

    with beam.Pipeline(argv=pipeline_args) as pipeline:
        with tft_beam.Context(temp_dir=working_dir):
            csv_coder = make_csv_coder(schema)
            raw_data = (
                pipeline
                | 'ReadFromText' >> beam.io.ReadFromText(
                  input_handle, skip_header_lines=1))
            decode_transform = beam.Map(csv_coder.decode)

            decoded_data = raw_data | 'DecodeForAnalyze' >> decode_transform
            transform_fn = (
                (decoded_data, raw_data_metadata) |
                ('Analyze' >> tft_beam.AnalyzeDataset(preprocessing_fn)))

            _ = (
                transform_fn
                | ('WriteTransformFn' >>
                   tft_beam.WriteTransformFn(working_dir)))


def _parse_line(line):
    """
    takes in a line of a csv file and returns its data as a feature dictionary
    :param line: the csv file's loaded line
    :return: the associated feature dictionary
    """
    fields = tf.decode_csv(
        line,
        record_defaults=_CSV_COLUMN_DEFAULTS,
        field_delim='|')
    features = {}
    for i in range(0, len(_CSV_COLUMNS_NAMES)):
        f = _CSV_COLUMNS_NAMES[i]
        if f in _UNUSED:
            continue
        features[f] = fields[i]
        labels = fields[5:7]
    labels = {}
    labels['lat'] = features['lat']
    labels['lon'] = features['lon']
    labels['LAT_LON_10'] = features['LAT_LON_10']
    features.pop('lat')
    features.pop('lon')
    features.pop('LAT_LON_10')

    labels['lat'] = (labels['lat'] - stats_dict['lat'][0]) / \
        stats_dict['lat'][1]
    labels['lon'] = (labels['lon'] - stats_dict['lon'][0]) / \
        stats_dict['lon'][1]

    return features, labels


def _concat(features):
    """
    :param features:
    :param features:
    :return:
    """
    columns = [
        'rsrp',
        'dt',
        'gci',
        'gci0',
        'gci1',
        'gci2',
        'rsrp0',
        'rsrp1',
        'rsrp2',
        'ta']
    final = {}
    for c in columns:
        final[c] = [features[c]]
        for i in range(1, window_size, 1):
            ftensor = [features['{}_{}'.format(c, i)]]
            final[c] = tf.concat([ftensor, final[c]], 0)
        if 'gci' not in c:
            # all features other than string are sparse
            final[c] = tf.contrib.layers.dense_to_sparse(
                final[c], eos_token=-2)
    return final


def _pre_process(line):
    """
    Overheads all csv processing functions.
    :param line: a raw csv line
    :return:
    """
    features, labels = _parse_line(line)
    features = _concat(features)
    return features, labels


def csv_input_fn(dataset_name, batch_size, num_epochs):
    """
    A predefined input function to feed an Estimator csv based cepidc files
    :param dataset_name: the file's ending type (either 'train, 'valid' or 'test')
    :param batch_size: the size of batches to feed the computational graph
    :param num_epochs: the number of time the entire dataset should be exposed to a gradient descent iteration
    :return: a BatchedDataset as a tuple of a feature dictionary and the labels
    """
    dataset = tf.data.TextLineDataset(dataset_name).skip(1)
    dataset = dataset.map(_pre_process)
   # dataset = dataset.batch(batch_size, drop_remainder=True)

    # TODO put shuffle back
 
    dataset = dataset.batch(
        batch_size).repeat(num_epochs)

    return dataset.prefetch(buffer_size=batch_size)


def normalize(x, c, stats_dict):
  #  if c == 'dt':
    #	return    x/120.0
    return x  # add_tensor(x,-stats_dict[c][0])/stats_dict[c][1]


def make_columns(transform_dir, filepath):
    """
    Builds the feature_columns required by the estimator to link the Dataset and the model_fn
    :return:
    """
    global stats_dict
    stats_dict = create_stats(transform_dir, filepath)
    vocab_file = os.path.join(transform_dir, 'tftransform_tmp/gci')
    columns_dict = {}

    columns_dict['gci'] = tf.feature_column.indicator_column(
        tf.feature_column.sequence_categorical_column_with_vocabulary_file(
            'gci',
            vocab_file,
            num_oov_buckets=5,
            # default_value="",
            dtype=tf.dtypes.string
        )
    )
    columns_dict['ta'] = (
        tf.feature_column.sequence_numeric_column(
            'ta', normalizer_fn=lambda x: normalize(x, 'ta', stats_dict)
        )
    )
    columns_dict['rsrp'] = (
        tf.feature_column.sequence_numeric_column(
            'rsrp', normalizer_fn=lambda x: normalize(x, 'rsrp', stats_dict)
        )
    )
    columns_dict['gci0'] = tf.feature_column.indicator_column(
        tf.feature_column.sequence_categorical_column_with_vocabulary_file(
            'gci0',
            vocab_file,
            num_oov_buckets=5,
            dtype=tf.dtypes.string
        )
    )
    columns_dict['gci1'] = tf.feature_column.indicator_column(
        tf.feature_column.sequence_categorical_column_with_vocabulary_file(
            'gci1',
            vocab_file,
            num_oov_buckets=5,
            dtype=tf.dtypes.string
        )
    )
    columns_dict['gci2'] = tf.feature_column.indicator_column(
        tf.feature_column.sequence_categorical_column_with_vocabulary_file(
            'gci2',
            vocab_file,
            num_oov_buckets=5,
            dtype=tf.dtypes.string
        )
    )
    columns_dict['rsrp0'] = (
        tf.feature_column.sequence_numeric_column(
            'rsrp0', normalizer_fn=lambda x: normalize(x, 'rsrp0', stats_dict)
        )
    )
    columns_dict['rsrp1'] = (
        tf.feature_column.sequence_numeric_column(
            'rsrp1', normalizer_fn=lambda x: normalize(x, 'rsrp1', stats_dict)
        )
    )
    columns_dict['rsrp2'] = (
        tf.feature_column.sequence_numeric_column(
            'rsrp2', normalizer_fn=lambda x: normalize(x, 'rsrp2', stats_dict)
        )
    )
    columns_dict['dt'] = (
        tf.feature_column.sequence_numeric_column(
            'dt', normalizer_fn=lambda x: normalize(x, 'dt', stats_dict)
        )
    )
    return columns_dict, stats_dict


def make_input_layers(dataset, feature_columns, batch_size):
    features = dataset[0]
    labels = dataset[1]
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
