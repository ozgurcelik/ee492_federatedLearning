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

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 500, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 20, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 10,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 64, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 64, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.01, 'Client learning rate.')

FLAGS = flags.FLAGS

tf.random.set_seed(42)
np.random.seed(42)
initializer = tf.keras.initializers.GlorotUniform(seed=42)


def create_wearable_dataset():
    ratio1 = 0.4
    ratio2 = 0.5
    ds = pd.read_csv('../../preponehot_actid_tab.csv', sep='\t')
    client_train_dataset = collections.OrderedDict()
    client_test_dataset = collections.OrderedDict()
    client_val_dataset = collections.OrderedDict()
    for i in range(len(ds.client_id.unique())):
        client_name = str(i)
        clientds = ds[ds.client_id == i+1]
        train, test1 = train_test_split(clientds, test_size=ratio1, random_state=42)
        test, val = train_test_split(test1, test_size=ratio2, random_state=42)
        data_train = collections.OrderedDict((('label', train.activity_id), ('measurements', train.iloc[:,range(3,len(train.columns)-1)])))
        data_test = collections.OrderedDict((('label', test.activity_id), ('measurements', test.iloc[:,range(3,len(test.columns)-1)])))
        data_val = collections.OrderedDict((('label', val.activity_id), ('measurements', val.iloc[:,range(3,len(val.columns)-1)])))
        client_train_dataset[client_name] = data_train
        client_test_dataset[client_name] = data_test
        client_val_dataset[client_name] = data_val
        
    train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
    test_dataset = tff.simulation.FromTensorSlicesClientData(client_test_dataset)
    val_dataset = tff.simulation.FromTensorSlicesClientData(client_val_dataset)

    return train_dataset, test_dataset, val_dataset

def preprocess(dataset):

    def batch_format_fn(element):
        return collections.OrderedDict(
                x=element['measurements'],
                y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(FLAGS.client_epochs_per_round).shuffle(1, seed=1).batch(
      FLAGS.batch_size).map(batch_format_fn)

def create_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(48,)),  # input shape required
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(9, activation=tf.nn.softmax)])

def server_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)

def client_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate)
  #return tf.keras.optimizers.Adam(learning_rate=FLAGS.client_learning_rate)



def personalized_eval(passed_server_state, testing, validating):
  opt = client_optimizer_fn()
  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  for client_num in range(len(testing)):
    client_model = create_keras_model()
    passed_server_state.model_weights.assign_weights_to(client_model)
    client_model_weights = tff.learning.ModelWeights.from_model(client_model)
    # if client_num%100 == 0:
    #   tf.print(client_model_weights.trainable[0][0][0])
    for batch in iter(testing[client_num]):
      with tf.GradientTape() as tape:
        preds = client_model(batch['x'], training=True)
        loss = loss_fn(batch['y'], preds)
      grads = tape.gradient(loss, client_model_weights.trainable)
      opt.apply_gradients(zip(grads, client_model.trainable_weights))
    for batch in validating[client_num]:
      preds = client_model(batch['x'], training=False)
      metric.update_state(y_true=batch['y'], y_pred=preds)
  return metric.result()


def evaluate(keras_model, validating):
  """Evaluate the acurracy of a keras model on a test dataset."""
  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  for client_num in range(len(validating)):
    for batch in validating[client_num]:
      predictions = keras_model(batch['x'])
      metric.update_state(y_true=batch['y'], y_pred=predictions)
  return metric.result()


def create_confusion_matrix(passed_server_state, testing, validating):
  y_true = []
  y_preds = []
  opt = client_optimizer_fn()
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  for client_num in range(len(testing)):
    client_model = create_keras_model()
    passed_server_state.model_weights.assign_weights_to(client_model)
    client_model_weights = tff.learning.ModelWeights.from_model(client_model)
    for batch in iter(testing[client_num]):
      with tf.GradientTape() as tape:
        preds = client_model(batch['x'], training=True)
        loss = loss_fn(batch['y'], preds)
      grads = tape.gradient(loss, client_model_weights.trainable)
      opt.apply_gradients(zip(grads, client_model.trainable_weights))
    for batch in validating[client_num]:
      y_true=np.concatenate([np.reshape(np.array(y_true),[-1,1]),np.reshape(np.array(batch['y']),[-1,1])])
      y_preds=np.concatenate([np.reshape(np.array(y_preds),[-1,1]),np.reshape(np.array(np.argmax(client_model(batch['x'], training=False), axis=1)),[-1,1])])
  cf_matrix = confusion_matrix(y_true, y_preds,normalize="true")
  return cf_matrix


def create_confusion_matrix2(keras_model, validating):
  y_true = []
  y_preds = []
  for client_num in range(len(validating)):
    for batch in validating[client_num]:
      # Optimize the model
      x = batch["x"]
      y = batch["y"]
      y_true=np.concatenate([np.reshape(np.array(y_true),[-1,1]),np.reshape(np.array(y),[-1,1])])
      y_preds=np.concatenate([np.reshape(np.array(y_preds),[-1,1]),np.reshape(np.array(np.argmax(keras_model(x, training=False), axis=1)),[-1,1])])
  cf_matrix = confusion_matrix(y_true, y_preds,normalize="true")
  return cf_matrix
    



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_data, test_data, val_data = create_wearable_dataset()
  test_set = [preprocess(test_data.create_tf_dataset_for_client(client)) for client in test_data.client_ids]
  val_set = [preprocess(val_data.create_tf_dataset_for_client(client)) for client in val_data.client_ids]


  def tff_model_fn():
    """Constructs a fully initialized model for use in federated averaging."""
    keras_model = create_keras_model()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    return tff.learning.from_keras_model(
        keras_model,
        loss=loss,
        metrics=metrics,
        input_spec=test_set[0].element_spec)

  iterative_process = simple_tff.build_federated_averaging_process(
      tff_model_fn, server_optimizer_fn, client_optimizer_fn)
  server_state = iterative_process.initialize()
  # Keras model that represents the global model we'll evaluate test data on.
  keras_model = create_keras_model()
  server_state.model_weights.assign_weights_to(keras_model)
  accuracy = personalized_eval(server_state, test_set, val_set)
  accuracy2 = evaluate(keras_model, val_set)
  print(f'\tInitial validation accuracy: {accuracy * 100.0:.2f}  accuracy 2: {accuracy2 * 100.0:.2f}%')


  for round_num in range(1,FLAGS.total_rounds+1):
    sampled_clients = np.random.choice(
        train_data.client_ids,
        size=FLAGS.train_clients_per_round,
        replace=False)
    sampled_train_data = [
        preprocess(train_data.create_tf_dataset_for_client(client))
        for client in sampled_clients
    ]
    server_state, train_metrics = iterative_process.next(
        server_state, sampled_train_data)
    #print(f'Round {round_num}')
    #print(f'\tTraining metrics: {train_metrics}')
    if round_num % FLAGS.rounds_per_eval == 0:
      server_state.model_weights.assign_weights_to(keras_model)
      accuracy = personalized_eval(server_state, test_set, val_set)
      accuracy2 = evaluate(keras_model, val_set)
      print(f'\tRound {round_num} validation accuracy: {accuracy * 100.0:.2f}  accuracy 2: {accuracy2 * 100.0:.2f}%')


  server_state.model_weights.assign_weights_to(keras_model)
  cf_matrix = create_confusion_matrix(server_state, test_set, val_set)
  hm = sn.heatmap(cf_matrix)
  fig = hm.get_figure()
  fig.savefig("personalized.jpg")

  cf_matrix2 = create_confusion_matrix2(keras_model, val_set)
  hm2 = sn.heatmap(cf_matrix2)
  fig2 = hm2.get_figure()
  fig2.savefig("normal.jpg")

  server_state.model_weights.assign_weights_to(keras_model)
  keras_model.save_weights('models/my_model_weights')


if __name__ == '__main__':
  app.run(main)