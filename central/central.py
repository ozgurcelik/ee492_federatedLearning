import numpy as np
from sklearn.metrics import confusion_matrix
import collections
import functools
from absl import app

import tensorflow as tf
import tensorflow_federated as tff

from stanfordDataset2 import get_stanford_federated_dataset
from stanfordDataset2 import preprocess
from resnet_models import create_resnet18

tf.random.set_seed(42)
np.random.seed(42)
initializer = tf.keras.initializers.GlorotUniform(seed=42)

import seaborn as sn
import matplotlib.pyplot as plt

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


def main(argv):

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_data, test_data = get_stanford_federated_dataset(size=32)
  train_set = preprocess(train_data.create_tf_dataset_from_all_clients(),batch_size=128)
  test_set = preprocess(test_data.create_tf_dataset_from_all_clients(),batch_size=128)
  #model = create_resnet18(input_shape=(64,64,3),num_classes=11,seed=42)
  model = create_original_fedavg_cnn_model(input_shape=[32,32,3])

  epochs = 40
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
  accuracy = evaluate(model,test_set)
  print(f'Initial validation accuracy: {accuracy * 100.0:.2f}%')
  f = open("central.txt", "a")
  f.write("\n")
  f.write(f'Initial validation accuracy: {accuracy * 100.0:.2f}%')
  f.write("\n")
  f.close()
  for epoch in range(1,epochs+1):
    for batch in iter(train_set):
      with tf.GradientTape() as tape:
        preds = model(batch['x'], training=True)
        loss = loss_object(batch['y'], preds)
      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
    if epoch%2==0:
      accuracy = evaluate(model,test_set)
      print(f'Round {epoch} validation accuracy: {accuracy*100:.2f}')
      f = open("central.txt", "a")
      f.write(f'Round {epoch} validation accuracy: {accuracy*100:.2f}')
      f.write("\n")
      f.close()

  cf_matrix = create_confusion_matrix(model, test_set)
  hm = sn.heatmap(cf_matrix)
  plt.show()
  fig = hm.get_figure()
  fig.savefig("central.jpg")


if __name__ == '__main__':
  app.run(main)

  
