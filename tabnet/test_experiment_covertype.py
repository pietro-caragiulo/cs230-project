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

"""Low-resource test for a small-scale TabNet experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
import data_helper_covertype
import numpy as np
import tabnet_model
import tensorflow as tf
import MakePlots as plot
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import pandas as pd
import csv

# Fix random seeds
np.random.seed(1)

def main(unused_argv):

  inFile = "combined_08mm"
  dataDir = "../notebooks/datasets"
  nEvents = 131072

  header_file = dataDir + "/" + inFile + "_header.csv"
  df = pd.read_csv(header_file)
  df_vz = df['vz']
  df_m = df['uncM']
  df_y = df['signal']

  # Load training and eval data.

  train_file = dataDir + "/" + inFile + "_train.csv"
  val_file = dataDir + "/" + inFile + "_val.csv"
  test_file = dataDir + "/" + inFile + "_test.csv"

  # TabNet model
  tabnet_forest_covertype = tabnet_model.TabNet(
      columns=data_helper_covertype.get_columns(),
      num_features=data_helper_covertype.num_features,
      feature_dim=128,
      output_dim=64,
      num_decision_steps=6,
      relaxation_factor=1.5,
      batch_momentum=0.7,
      virtual_batch_size=512,
      num_classes=data_helper_covertype.num_classes)

  column_names = sorted(data_helper_covertype.feature_columns)
  print(
      "Ordered column names, corresponding to the indexing in Tensorboard visualization"
  )
  for fi in range(len(column_names)):
    print(str(fi) + " : " + column_names[fi])

  # Training parameters
  max_steps = 10
  display_step = 1
  val_step = 5
  save_step = 5
  init_localearning_rate = 0.02
  decay_every = 500
  decay_rate = 0.95
  batch_size = nEvents
  sparsity_loss_weight = 0.0001
  gradient_thresh = 2000.0
  nSamples = nEvents
  min_steps = max_steps - int(nSamples/batch_size)

  # Input sampling
  train_batch = data_helper_covertype.input_fn(
      train_file,
      num_epochs=100000,
      shuffle=True,
      batch_size=batch_size,
      n_buffer=1,
      n_parallel=1)
  val_batch = data_helper_covertype.input_fn(
      val_file,
      num_epochs=10000,
      shuffle=False,
      batch_size=batch_size,
      n_buffer=1,
      n_parallel=1)
  test_batch = data_helper_covertype.input_fn(
      test_file,
      num_epochs=10000,
      shuffle=False,
      batch_size=32768,
      n_buffer=1,
      n_parallel=1)

  train_iter = train_batch.make_initializable_iterator()
  val_iter = val_batch.make_initializable_iterator()
  test_iter = test_batch.make_initializable_iterator()

  feature_train_batch, label_train_batch = train_iter.get_next()
  feature_val_batch, label_val_batch = val_iter.get_next()
  feature_test_batch, label_test_batch = test_iter.get_next()

  #feature_test_batch = feature_test_batch2

  # Define the model and losses

  encoded_train_batch, total_entropy = tabnet_forest_covertype.encoder(
      feature_train_batch, reuse=False, is_training=True)

  logits_orig_batch, prediction_train = tabnet_forest_covertype.classify(
      encoded_train_batch, reuse=False)
  softmax_orig_key_op = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits_orig_batch, labels=label_train_batch))

  train_loss_op = softmax_orig_key_op + sparsity_loss_weight * total_entropy
  tf.summary.scalar("Total_loss_test", train_loss_op)

  # Optimization step
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(
      init_localearning_rate,
      global_step=global_step,
      decay_steps=decay_every,
      decay_rate=decay_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    gvs = optimizer.compute_gradients(train_loss_op)
    capped_gvs = [(tf.clip_by_value(grad, -gradient_thresh,
                                    gradient_thresh), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

  # Model evaluation

  # Validation performance
  encoded_val_batch, total_entropy_val = tabnet_forest_covertype.encoder(
      feature_val_batch, reuse=True, is_training=True)

  logits_val_batch, prediction_val = tabnet_forest_covertype.classify(
      encoded_val_batch, reuse=True)

  softmax_val_key_op = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits_val_batch, labels=label_val_batch))

  val_loss_op = softmax_val_key_op + sparsity_loss_weight * total_entropy_val
  tf.summary.scalar("Total_loss_val", val_loss_op)

  predicted_labels = tf.cast(tf.argmax(prediction_val, 1), dtype=tf.int32)
  val_eq_op = tf.equal(predicted_labels, label_val_batch)
  val_acc_op = tf.reduce_mean(tf.cast(val_eq_op, dtype=tf.float32))
  tf.summary.scalar("Val_accuracy", val_acc_op)

  # Test performance
  encoded_test_batch, _ = tabnet_forest_covertype.encoder(
      feature_test_batch, reuse=True, is_training=True)

  _, prediction_test = tabnet_forest_covertype.classify(
      encoded_test_batch, reuse=True)

  predicted_labels = tf.cast(tf.argmax(prediction_test, 1), dtype=tf.int32)
  test_eq_op = tf.equal(predicted_labels, label_test_batch)
  test_acc_op = tf.reduce_mean(tf.cast(test_eq_op, dtype=tf.float32))
  tf.summary.scalar("Test_accuracy", test_acc_op)

  # Training setup
  model_name = "tabnet_forest_covertype_model_" + inFile
  init = tf.initialize_all_variables()
  init_local = tf.local_variables_initializer()
  init_table = tf.tables_initializer(name="Initialize_all_tables")
  saver = tf.train.Saver()
  summaries = tf.summary.merge_all()

  with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("./tflog/" + model_name, sess.graph)

    sess.run(init)
    sess.run(init_local)
    sess.run(init_table)
    sess.run(train_iter.initializer)
    sess.run(val_iter.initializer)
    sess.run(test_iter.initializer)

    for step in range(1, max_steps + 1):
      if step % display_step == 0:
        _, train_loss, merged_summary = sess.run(
            [train_op, train_loss_op, summaries])
        summary_writer.add_summary(merged_summary, step)
        print("Step " + str(step) + " , Training Loss = " +
              "{:.4f}".format(train_loss))
      else:
        _ = sess.run(train_op)

      if step % val_step == 0:
        feed_arr = [
            vars()["summaries"],
            vars()["val_acc_op"],
            vars()["test_acc_op"]
        ]

        val_arr = sess.run(feed_arr)
        merged_summary = val_arr[0]
        val_acc = val_arr[1]

        print("Step " + str(step) + " , Val Accuracy = " +
              "{:.4f}".format(val_acc))
        summary_writer.add_summary(merged_summary, step)

      if step % save_step == 0:
        saver.save(sess, "./checkpoints/" + model_name + ".ckpt")

      if(step == max_steps):
        with open("./tflog/" + model_name + '/' + inFile + '_out.csv', mode='w') as output_file:
          file_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          file_writer.writerow(["ex","yhat","y","vz","uncM"])
          yhat = prediction_test[:,1].eval()
          type(yhat)
          z = feature_test_batch['vz'].eval()
          y = label_test_batch.eval()
          m = feature_test_batch['uncM'].eval()
          for i in range(len(y)):
            file_writer.writerow([str(i),str(yhat[i]),str(y[i]),str(z[i]),str(m[i])])

      #if(step > min_steps):
        #j = step
        #with open("./tflog/" + model_name + '/' + inFile + '_out.csv', mode='w') as output_file:
          #file_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          #file_writer.writerow(["ex","yhat","y","vz","uncM"])
          #yhat = prediction_test[:,1].eval()
          #y = label_test_batch.eval()
          #for i in range(batch_size):
            #index = (j - min_steps - 1) * batch_size + i
            #tf.summary.scalar("ytest_hat_%d_%d" % (j, i), prediction_test[i][1])
            #tf.summary.scalar("ytest_%d_%d" % (j, i), label_test_batch[i])
            #tf.summary.scalar("yval_hat_%d_%d" % (j, i), prediction_val[i][1])
            #tf.summary.scalar("yval_%d_%d" % (j, i), label_val_batch[i])
            #tf.summary.scalar("ytrain_hat_%d_%d" % (j, i), prediction_train[i][1])
            #tf.summary.scalar("ytrain_%d_%d" % (j, i), label_train_batch[i])
            #tf.summary.scalar("vztrain_%d_%d" % (j, i), feature_train_batch['vz'][i])
            #tf.summary.scalar("vzval_%d_%d" % (j, i), feature_val_batch['vz'][i])
            #tf.summary.scalar("vztest_%d_%d" % (j, i), df_vz[index])
            #tf.summary.scalar("uncmtrain_%d_%d" % (j, i), feature_train_batch['uncM'][i])
            #tf.summary.scalar("uncmval_%d_%d" % (j, i), feature_val_batch['uncM'][i])
            #tf.summary.scalar("uncmtest_%d_%d" % (j, i), df_m[index])
            #file_writer.writerow([str(index),str(yhat[i]),str(y[i]),str(df_vz[index]),str(df_m[index])])

if __name__ == "__main__":
  app.run(main)
