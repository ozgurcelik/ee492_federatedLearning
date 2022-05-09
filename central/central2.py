import numpy as np
from sklearn.metrics import confusion_matrix
import collections
from absl import app

import tensorflow as tf
import tensorflow_federated as tff

from stanfordDataset import get_stanford_federated_dataset
from stanfordDataset import preprocess
from resnet_models import create_resnet18

tf.random.set_seed(42)
np.random.seed(42)

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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_data, test_data = get_stanford_federated_dataset
  train_set = [preprocess(train_data.create_tf_dataset_for_client(x)) for x in train_data.client_ids]
  test_set = [preprocess(test_data.create_tf_dataset_for_client(x)) for x in test_data.client_ids]
  model = create_resnet18(input_shape=(112,112,3),num_classes=11,seed=42)

  epochs = 40
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
  for epoch in range(epochs):
    print(epoch)
    for batch in train_data:
	  	with tf.GradientTape() as tape:
        preds = client_model(batch['x'], training=True)
        loss = loss_fn(batch['y'], preds)
      grads = tape.gradient(loss, client_model_weights.trainable)
      opt.apply_gradients(zip(grads, client_model.trainable_weights))
  
