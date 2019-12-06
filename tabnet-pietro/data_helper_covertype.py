# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data helper function for the Forest Covertype dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Dataset size
n_train_samples = 309871
n_val_samples = 154937
n_test_samples = 116203
num_features = 10
num_classes = 2

# All feature columns in the data
label_column = "signal"

bool_columns = []

int_columns = []

str_columns = []
str_nuniques = []
#float_columns = ["truthZ","vx","vy","vz","vxPull","vyPull","vzPull","uncP","uncChisq","uncM","projX","projY","projXPull","projYPull","bscChisq","tarChisq","eleP","eleChisqDOF","eleTrkD0","eleTrkTanLambda","eleZ0","eleTrkD0Err","eleTrkTanLambdaErr","eleTrkZ0Err","posP","posChisqDOF","posTrkD0","posTrkTanLambda","posZ0","posTrkD0Err","posTrkTanLambdaErr","posTrkZ0Err"]
float_columns = ['vz','vzPull','vy','vyPull', 'uncM', 'eleZ0', 'posZ0', 'projY', 'eleTrkTanLambda','posTrkTanLambda']

feature_columns = (
    int_columns + bool_columns + str_columns + float_columns + [label_column])
all_columns = feature_columns + [label_column]
defaults = ([[0] for col in int_columns] + [[""] for col in bool_columns] +
            [[0.0] for col in float_columns] + [[""] for col in str_columns] +
            [[-1]])


def get_columns():
  """Get the representations for all input columns."""

  columns = []
  if float_columns:
    columns += [tf.feature_column.numeric_column(ci) for ci in float_columns]
  if int_columns:
    columns += [tf.feature_column.numeric_column(ci) for ci in int_columns]
  if str_columns:
    # pylint: disable=g-complex-comprehension
    columns += [
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                ci, hash_bucket_size=int(3 * num)),
            dimension=1) for ci, num in zip(str_columns, str_nuniques)
    ]
  if bool_columns:
    # pylint: disable=g-complex-comprehension
    columns += [
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                ci, hash_bucket_size=3),
            dimension=1) for ci in bool_columns
    ]
  return columns


def input_fn(data_file,
             num_epochs,
             shuffle,
             batch_size,
             n_buffer=50,
             n_parallel=16):
  """Function to read the input file and return the dataset."""

  def parse_csv(value_column):
    columns = tf.decode_csv(value_column, record_defaults=defaults)
    features = dict(zip(all_columns, columns))
    print(features)
    label = features.pop(label_column)
    classes = tf.cast(label, tf.int32)# - 1
    return features, classes

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=n_buffer)

  dataset = dataset.map(parse_csv, num_parallel_calls=n_parallel)

  # Repeat after shuffling, to prevent separate epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset
