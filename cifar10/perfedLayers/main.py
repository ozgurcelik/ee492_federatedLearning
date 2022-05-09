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
import simple_tf
from cifar10_dataset import get_federated_datasets

# Training hyperparameters
flags.DEFINE_integer('train_clients_per_round', 10, 'How many clients to sample per round.')
flags.DEFINE_integer('total_num_clients', 10, 'Number of total clients for dataset creation.')
flags.DEFINE_integer('total_rounds', 80, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 2, 'How often to evaluate')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 128, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.01, 'Client learning rate.')

FLAGS = flags.FLAGS

tf.random.set_seed(42)
np.random.seed(42)
initializer = tf.keras.initializers.GlorotUniform(seed=42)

def create_original_fedavg_cnn_model(only_digits=True):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  input_shape = [32, 32, 3]

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
      tf.keras.layers.Dense(10 if only_digits else 62, kernel_initializer=initializer),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def personalized_eval(passed_server_state, client_states, testing):
  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
  for client_num in range(len(testing)):
    client_model = create_original_fedavg_cnn_model()
    passed_server_state.model_weights.assign_weights_to(client_model)
    client_model_weights = tff.learning.ModelWeights.from_model(client_model)
    client_model_weights.trainable[-1].assign(client_states[client_num].perFed_layers.trainable[-1])
    client_model_weights.trainable[-2].assign(client_states[client_num].perFed_layers.trainable[-2])
    client_model_weights.trainable[-3].assign(client_states[client_num].perFed_layers.trainable[-3])
    client_model_weights.trainable[-4].assign(client_states[client_num].perFed_layers.trainable[-4])
    for batch in testing[client_num]:
      preds = client_model(batch['x'], training=False)
      metric.update_state(y_true=batch['y'], y_pred=preds)
  return metric.result()


def create_personalized_confusion_matrix(passed_server_state, client_states, testing):
  y_true = []
  y_preds = []
  for client_num in range(len(testing)):
    client_model = create_original_fedavg_cnn_model()
    passed_server_state.model_weights.assign_weights_to(client_model)
    client_model_weights = tff.learning.ModelWeights.from_model(client_model)
    client_model_weights.trainable[-1].assign(client_states[client_num].perFed_layers.trainable[-1])
    client_model_weights.trainable[-2].assign(client_states[client_num].perFed_layers.trainable[-2])
    client_model_weights.trainable[-3].assign(client_states[client_num].perFed_layers.trainable[-3])
    client_model_weights.trainable[-4].assign(client_states[client_num].perFed_layers.trainable[-4])
    for batch in testing[client_num]:
      x = batch["x"]
      y = batch["y"]
      y_true=np.concatenate([np.reshape(np.array(y_true),[-1,1]),np.reshape(np.array(y),[-1,1])])
      y_preds=np.concatenate([np.reshape(np.array(y_preds),[-1,1]),np.reshape(np.array(np.argmax(client_model(x, training=False), axis=1)),[-1,1])])
  cf_matrix = confusion_matrix(y_true, y_preds,normalize="true")
  return cf_matrix



def server_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate, momentum=0.9)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_data, test_data = get_federated_datasets(num_clients=FLAGS.total_num_clients, train_client_batch_size=FLAGS.batch_size)
  test_dataset = [test_data.create_tf_dataset_for_client(client) for client in test_data.client_ids]

  def tff_model_fn():
    """Constructs a fully initialized model for use in federated averaging."""
    keras_model = create_original_fedavg_cnn_model(only_digits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    return tff.learning.from_keras_model(
        keras_model,
        loss=loss,
        metrics=metrics,
        input_spec=test_dataset[0].element_spec)


  layers = [-4,-3,-2,-1]
  def get_sample_client_state():
    # Return a sample client state to initialize TFF types.
    return simple_tf.ClientState(client_index=-1,
      perFed_layers=simple_tf.ModelWeights(trainable= [tff_model_fn().trainable_variables[l] for l in layers]
      ))
  
  client_states = [simple_tf.ClientState(client_index=i,
      perFed_layers=simple_tf.ModelWeights(trainable= [tff_model_fn().trainable_variables[l] for l in layers]
      )) for i in train_data.client_ids]

  iterative_process = simple_tff.build_federated_averaging_process(
      tff_model_fn, get_sample_client_state, server_optimizer_fn, client_optimizer_fn)
  server_state = iterative_process.initialize()
  # Keras model that represents the global model we'll evaluate test data on.
  keras_model = create_original_fedavg_cnn_model(only_digits=True)
  accuracy = personalized_eval(server_state, client_states, test_dataset)
  print(f'\tInitial validation accuracy: {accuracy * 100.0:.2f}%')

  f = open("10cnoniid.txt", "a")
  f.write("\n")
  f.write(f'Initial validation accuracy: {accuracy * 100.0:.2f}%')
  f.write("\n")
  f.close()

  for round_num in range(1,FLAGS.total_rounds+1):

    sampled_clients = np.random.choice(
        train_data.client_ids,
        size=FLAGS.train_clients_per_round,
        replace=False)
    sampled_train_data = [
        train_data.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]
    sampled_client_states = [
        client_states[client] for client in sampled_clients
    ]  # Sample corresponding client states.
    server_state, train_metrics , updated_client_states= iterative_process.next(
        server_state, sampled_train_data, sampled_client_states)
    for client_state in updated_client_states:
      client_id = train_data.client_ids[client_state.client_index]
      client_states[client_id] = client_state
    if round_num % FLAGS.rounds_per_eval == 0:
      f = open("10cnoniid.txt", "a")
      accuracy = personalized_eval(server_state, client_states, test_dataset)
      print(f'\tRound {round_num} validation accuracy: {accuracy * 100.0:.2f}%')
      f.write(f'Round {round_num} validation accuracy: {accuracy*100:.2f}')
      f.write("\n")
      f.close()
    

  cf_matrix = create_personalized_confusion_matrix(server_state, client_states, test_dataset)
  hm = sn.heatmap(cf_matrix)
  plt.show()
  fig = hm.get_figure()
  fig.savefig("10cnoniid.jpg")


if __name__ == '__main__':
  app.run(main)