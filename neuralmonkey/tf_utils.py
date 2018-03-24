"""A set of helper functions for TensorFlow."""
from typing import Callable, Iterable, List, Optional, Tuple
# pylint: disable=unused-import
from typing import Dict, Set
# pylint: enable=unused-import
import tensorflow as tf


def _get_current_experiment():
    # This is needed to avoid circular imports.
    from neuralmonkey.experiment import Experiment
    return Experiment.get_current()


def update_initializers(initializers: Iterable[Tuple[str, Callable]]) -> None:
    _get_current_experiment().update_initializers(initializers)


def get_initializer(var_name: str,
                    default: Callable = None) -> Optional[Callable]:
    """Return the initializer associated with the given variable name.

    The name of the current variable scope is prepended to the variable name.

    This should only be called during model building.
    """
    full_name = tf.get_variable_scope().name + "/" + var_name
    return _get_current_experiment().get_initializer(full_name, default)


def get_variable(name: str,
                 shape: List[Optional[int]] = None,
                 dtype: tf.DType = None,
                 initializer: Callable = None,
                 **kwargs) -> tf.Variable:
    """Get an existing variable with these parameters or create a new one.

    This is a wrapper around `tf.get_variable`. The `initializer` parameter is
    treated as a default which can be overriden by a call to
    `update_initializers`.

    This should only be called during model building.
    """
    return tf.get_variable(
        name=name, shape=shape, dtype=dtype,
        initializer=get_initializer(name, initializer),
        **kwargs)


def layer_norm(x, epsilon=1e-6):
    """Layer normalize the tensor x, averaging over the last dimension.

    Implementation based on tensor2tensor.
    """
    with tf.variable_scope("LayerNorm"):
        gamma = get_variable(
            name="gamma",
            shape=[x.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        beta = get_variable(
            name="beta",
            shape=[x.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(
            tf.square(x - mean),
            axis=[-1],
            keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * gamma + beta
