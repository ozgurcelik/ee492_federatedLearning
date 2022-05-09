# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple FedAvg to train EMNIST.

This is intended to be a minimal stand-alone experiment script demonstrating
usage of TFF's Federated Compute API for a from-scratch Federated Avearging
implementation.
"""

import collections
import functools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

import seaborn as sn
import matplotlib.pyplot as plt

import simple_tff

from new_stanfordDataset import get_stanford_federated_dataset
from new_stanfordDataset import preprocess
from resnet_models import create_resnet18

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 300, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 5, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 6,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 10, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 128, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.01, 'Client learning rate.')

FLAGS = flags.FLAGS

tf.random.set_seed(42)
np.random.seed(42)
initializer = tf.keras.initializers.GlorotUniform(seed=42)


def evaluate(keras_model, test_dataset):
  """Evaluate the acurracy of a keras model on a test dataset."""
  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  for batch in test_dataset:
    predictions = keras_model(batch['x'])
    metric.update_state(y_true=batch['y'], y_pred=predictions)
  return metric.result()


def create_confusion_matrix(keras_model, test_dataset):
  y_true = []
  y_preds = []
  for batch in test_dataset:
    # Optimize the model
    x = batch["x"]
    y = batch["y"]
    y_true=np.concatenate([np.reshape(np.array(y_true),[-1,1]),np.reshape(np.array(y),[-1,1])])
    y_preds=np.concatenate([np.reshape(np.array(y_preds),[-1,1]),np.reshape(np.array(np.argmax(keras_model(x, training=False), axis=1)),[-1,1])])
  cf_matrix = confusion_matrix(y_true, y_preds,normalize="true")
  return cf_matrix

def create_original_fedavg_cnn_model(input_shape = [32, 32, 3]):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
      kernel_initializer=initializer)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(11),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model



def create_original_fedavg_cnn_model2(input_shape = [64, 64, 3]):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
      kernel_initializer=initializer)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(11),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def server_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate, momentum=0.9)
  #return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate)


size = (64 ,64)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_data, test_data = get_stanford_federated_dataset(size=size,TRAIN_EXAMPLES_PER_LABEL=192,dirichlet_parameter=0.7,num_clients=6)
  #train_data = get_stanford_federated_dataset(size=(32,32),TRAIN_EXAMPLES_PER_LABEL=180,dirichlet_parameter=0.2,num_clients=3)
  train_set = [preprocess(train_data.create_tf_dataset_for_client(client),batch_size=FLAGS.batch_size) for client in train_data.client_ids]
  test_set = preprocess(test_data.create_tf_dataset_from_all_clients(),batch_size=128)
  #test_set = preprocess(train_data.create_tf_dataset_from_all_clients(),batch_size=128)

  def tff_model_fn():
    """Constructs a fully initialized model for use in federated averaging."""
    keras_model = create_original_fedavg_cnn_model(input_shape=[size[0],size[1],3])
    #keras_model = create_original_fedavg_cnn_model2(input_shape=[size[0],size[1],3])
    #keras_model = create_resnet18(input_shape=(size[0],size[1],3),num_classes=11,seed=42)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    return tff.learning.from_keras_model(
        keras_model,
        loss=loss,
        metrics=metrics,
        input_spec=test_set.element_spec)

  iterative_process = simple_tff.build_federated_averaging_process(
      tff_model_fn, server_optimizer_fn, client_optimizer_fn)
  server_state = iterative_process.initialize()
  # Keras model that represents the global model we'll evaluate test data on.
  keras_model = create_original_fedavg_cnn_model(input_shape=[size[0],size[1],3])
  #keras_model = create_original_fedavg_cnn_model2(input_shape=[size[0],size[1],3])
  #keras_model = create_resnet18(input_shape=(size[0],size[1],3),num_classes=11,seed=42)
  server_state.model_weights.assign_weights_to(keras_model)
  accuracy = evaluate(keras_model, test_set)
  print(f'\tInitial validation accuracy: {accuracy * 100.0:.2f}%')
  f = open("baselineFed.txt", "a")
  f.write("\n")
  f.write(f'Initial validation accuracy: {accuracy * 100.0:.2f}%')
  f.write("\n")
  f.close()


  for round_num in range(1,FLAGS.total_rounds+1):
    # sampled_clients = np.random.choice(
    #     train_data.client_ids,
    #     size=FLAGS.train_clients_per_round,
    #     replace=False)
    # sampled_train_data = [train_set[x] for x in sampled_clients]
    server_state, train_metrics = iterative_process.next(
        server_state, train_set)
    #print(f'Round {round_num}')
    #print(f'\tTraining metrics: {train_metrics}')
    if round_num % FLAGS.rounds_per_eval == 0:
      server_state.model_weights.assign_weights_to(keras_model)
      accuracy = evaluate(keras_model, test_set)
      print(f'\tRound {round_num} validation accuracy: {accuracy * 100.0:.2f}%')
      f = open("baselineFed.txt", "a")
      f.write(f'Round {round_num} validation accuracy: {accuracy*100:.2f}')
      f.write("\n")
      f.close()


  server_state.model_weights.assign_weights_to(keras_model)
  cf_matrix = create_confusion_matrix(keras_model, test_set)
  hm = sn.heatmap(cf_matrix)
  plt.show()
  fig = hm.get_figure()
  fig.savefig("50client200roundexplossweighted.jpg")


if __name__ == '__main__':
  app.run(main)