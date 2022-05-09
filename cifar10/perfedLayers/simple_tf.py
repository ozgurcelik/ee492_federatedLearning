import collections
from typing import Union

import attr
import tensorflow as tf
import tensorflow_federated as tff

ModelWeights = collections.namedtuple('ModelWeights', 'trainable')
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')

@attr.s
class ModelOutputs:
  """A container of local client training outputs."""
  loss = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.
  Attributes:
    weights_delta: A dictionary of updates to the model's trainable variables.
    client_weight: Weight to be used in a weighted mean when aggregating
      `weights_delta`.
    model_output: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  client_state = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ClientState(object):
  """Structure for state on the client.

  Fields:
  -   `client_index`: The client index integer to map the client state back to
      the database hosting client states in the driver file.
  -   `iters_count`: The number of total iterations a client has computed in
      the total rounds so far.
  """
  client_index = attr.ib()
  perFed_layers = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
  """Structure for state on the server.
  Attributes:
    model_weights: A dictionary of model's trainable variables.
    optimizer_state: Variables of optimizer.
    round_num: The current round in the training process.
  """
  model_weights = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
  """Structure for tensors broadcasted by server during federated optimization.
  Attributes:
    model_weights: A dictionary of model's trainable tensors.
    round_num: Round index to broadcast. We use `round_num` as an example to
      show how to broadcast auxiliary information that can be helpful on
      clients. It is not explicitly used, but can be applied to enable learning
      rate scheduling.
  """
  model_weights = attr.ib()
  round_num = attr.ib()


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
  """Updates `server_state` based on `weights_delta`.
  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
      creates variables, they must have already been created.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.
  Returns:
    An updated `ServerState`.
  """
  # Initialize the model with the current state.
  model_weights = tff.learning.ModelWeights.from_model(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        server_state.model_weights)
  tf.nest.map_structure(lambda v, t: v.assign(t), server_optimizer.variables(),
                        server_state.optimizer_state)

  # Apply the update to the model.
  neg_weights_delta = [-1.0 * x for x in weights_delta]
  server_optimizer.apply_gradients(
      zip(neg_weights_delta, model_weights.trainable), name='server_update')

  # Create a new state based on the updated model.
  return ServerState(
      model_weights=model_weights,
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1)


@tf.function
def build_server_broadcast_message(server_state):
  """Builds `BroadcastMessage` for broadcasting.
  This method can be used to post-process `ServerState` before broadcasting.
  For example, perform model compression on `ServerState` to obtain a compressed
  state that is sent in a `BroadcastMessage`.
  Args:
    server_state: A `ServerState`.
  Returns:
    A `BroadcastMessage`.
  """
  return BroadcastMessage(
      model_weights=server_state.model_weights,
      round_num=server_state.round_num)

def get_client_state(old_client_state, model_weights):
  layers = [-4,-3,-2,-1]
  new_client_state = ClientState(client_index=old_client_state.client_index,
      perFed_layers=ModelWeights(trainable= [model_weights.trainable[l] for l in layers]
      ))
  return new_client_state



@tf.function
def client_update(model, dataset, client_state, server_message, client_optimizer):
  """Performans client local training of `model` on `dataset`.
  Args:
    model: A `tff.learning.Model` to train locally on the client.
    dataset: A 'tf.data.Dataset' representing the clients local dataset.
    server_message: A `BroadcastMessage` from serve containing the initial
      model weights to train.
    client_optimizer: A `tf.keras.optimizers.Optimizer` used to update the local
      model during training.
  Returns:
    A `ClientOutput` instance with a model update to aggregate on the server.
  """
  model_weights = tff.learning.ModelWeights.from_model(model)
  initial_weights = server_message.model_weights
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        initial_weights)

  model_weights.trainable[-1].assign(client_state.perFed_layers.trainable[-1])
  model_weights.trainable[-2].assign(client_state.perFed_layers.trainable[-2])
  model_weights.trainable[-3].assign(client_state.perFed_layers.trainable[-3])
  model_weights.trainable[-4].assign(client_state.perFed_layers.trainable[-4])

  num_examples = tf.constant(0, dtype=tf.int32)
  loss_sum = tf.constant(0, dtype=tf.float32)


  for batch in iter(dataset):
    with tf.GradientTape() as tape:
      outputs = model.forward_pass(batch)
    grads = tape.gradient(outputs.loss, model_weights.trainable)
    client_optimizer.apply_gradients(zip(grads, model_weights.trainable))
    batch_size = tf.shape(batch['y'])[0]
    num_examples += batch_size
    loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)

  new_client_state = get_client_state(client_state, model_weights)

  weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                        model_weights.trainable,
                                        initial_weights.trainable)
  client_weight = tf.cast(num_examples, tf.float32)
  return ClientOutput(weights_delta, client_weight, loss_sum / client_weight, new_client_state)