"""Implementation of the encoder of the Transformer model.

Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762
"""
# pylint: disable=unused-import
from typing import Set, Optional, List
# pylint: enable=unused-import

import math
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.attention.scaled_dot_product import attention
from neuralmonkey.logging import log
from neuralmonkey.model.model_part import FeedDict, ModelPart
from neuralmonkey.model.stateful import (TemporalStateful,
                                         TemporalStatefulWithOutput)
from neuralmonkey.nn.utils import dropout
from neuralmonkey.tf_utils import get_variable, layer_norm


def position_signal(dimension: int, length: tf.Tensor) -> tf.Tensor:
    # Code simplified and copied from github.com/tensorflow/tensor2tensor

    # TODO write this down on a piece of paper and understand the code and
    # compare it to the paper
    positions = tf.to_float(tf.range(length))

    num_timescales = dimension // 2

    # see: github.com/tensorflow/tensor2tensor/blob/v1.5.5/tensor2tensor/
    #      layers/common_attention.py#L425
    log_timescale_increment = math.log(1.0e4) / (num_timescales - 1)
    inv_timescales = tf.exp(tf.range(num_timescales, dtype=tf.float32)
                            * -log_timescale_increment)

    scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(
        inv_timescales, 0)

    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(dimension, 2)]])
    signal = tf.reshape(signal, [1, length, dimension])

    return signal


class TransformerLayer(TemporalStateful):
    def __init__(self, states: tf.Tensor, mask: tf.Tensor) -> None:
        self._states = states
        self._mask = mask

    @property
    def temporal_states(self) -> tf.Tensor:
        return self._states

    @property
    def temporal_mask(self) -> tf.Tensor:
        return self._mask


class TransformerEncoder(ModelPart, TemporalStatefulWithOutput):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: TemporalStateful,
                 ff_hidden_size: int,
                 depth: int,
                 n_heads: int,
                 dropout_keep_prob: float = 1.0,
                 attention_dropout_keep_prob: float = 1.0,
                 target_space_id: int = None,
                 use_att_transform_bias: bool = False,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create an encoder of the Transformer model.

        Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762

        Arguments:
            input_sequence: Embedded input sequence.
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects.
            dropout_keep_prob: Probability of keeping a value during dropout.

        Keyword arguments:
            ff_hidden_size: Size of the feedforward sublayers.
            n_heads: Number of the self-attention heads.
            depth: Number of sublayers.
            attention_dropout_keep_prob: Probability of keeping a value
                during dropout on the attention output.
        """
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.input_sequence = input_sequence
        self.model_dimension = self.input_sequence.dimension
        self.ff_hidden_size = ff_hidden_size
        self.depth = depth
        self.n_heads = n_heads
        self.dropout_keep_prob = dropout_keep_prob
        self.attention_dropout_keep_prob = attention_dropout_keep_prob
        self.target_space_id = target_space_id
        self.use_att_transform_bias = use_att_transform_bias

        if self.depth <= 0:
            raise ValueError("Depth must be a positive integer.")

        if self.ff_hidden_size <= 0:
            raise ValueError("Feed forward hidden size must be a "
                             "positive integer.")

        if self.dropout_keep_prob <= 0.0 or self.dropout_keep_prob > 1.0:
            raise ValueError("Dropout keep prob must be inside (0,1].")

        if (self.attention_dropout_keep_prob <= 0.0
                or self.attention_dropout_keep_prob > 1.0):
            raise ValueError("Dropout keep prob for attn must be in (0,1].")

        if self.target_space_id is not None and (self.target_space_id >= 32
                                                 or self.target_space_id < 0):
            raise ValueError(
                "If provided, the target space ID should be between 0 and 31. "
                "Was: {}".format(self.target_space_id))

        self.train_mode = tf.placeholder(tf.bool, [], "train_mode")
        log("Output op: {}".format(self.output))
    # pylint: enable=too-many-arguments

    @tensor
    def output(self) -> tf.Tensor:
        return tf.reduce_sum(self.temporal_states, axis=1)

    @tensor
    def modality_matrix(self) -> tf.Tensor:
        """Create an embedding matrix for varyining target modalities.

        Used to embed different target space modalities in the tensor2tensor
        models (e.g. during the zero-shot translation).

        TODO:
            Move this to the model.sequence as an optional network parameter.
        """
        emb_size = self.input_sequence.temporal_states.shape.as_list()[-1]
        return get_variable(
            name="target_modality_embedding_matrix",
            shape=[32, emb_size],
            dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer())

    @tensor
    def target_modality_embedding(self) -> tf.Tensor:
        """Gather correct embedding of the target space modality.

        See TransformerEncoder.modality_matrix for more information.

        TODO:
            Make a target_space_id an optional parameter of the experiment.
            Move it either to a runner or experiment configuration.
        """

        # TODO: add dropout?
        return tf.gather(self.modality_matrix,
                         tf.constant(self.target_space_id))

    @tensor
    def encoder_inputs(self) -> tf.Tensor:
        inputs = self.input_sequence.temporal_states

        if self.target_space_id is not None:
            inputs += tf.reshape(self.target_modality_embedding, [1, 1, -1])

        length = tf.shape(inputs)[1]
        pos = position_signal(self.model_dimension, length)

        return dropout(inputs + pos, self.dropout_keep_prob, self.train_mode)

    def self_attention_sublayer(
            self, prev_layer: TransformerLayer) -> tf.Tensor:
        # Layer normalization
        normalized_states = layer_norm(prev_layer.temporal_states)

        # Run self-attention
        self_context, _ = attention(
            queries=normalized_states,
            keys=normalized_states,
            values=normalized_states,
            keys_mask=prev_layer.temporal_mask,
            num_heads=self.n_heads,
            dropout_callback=lambda x: dropout(
                x, self.attention_dropout_keep_prob, self.train_mode),
            use_bias=self.use_att_transform_bias)

        # Apply dropout
        self_context = dropout(
            self_context, self.dropout_keep_prob, self.train_mode)

        # Add residual connections
        return self_context + prev_layer.temporal_states

    def feedforward_sublayer(self, layer_input: tf.Tensor) -> tf.Tensor:
        # Layer normalization
        normalized_input = layer_norm(layer_input)

        # Feed-forward network hidden layer + ReLU
        ff_hidden = tf.layers.dense(
            normalized_input, self.ff_hidden_size, activation=tf.nn.relu,
            name="hidden_state")

        # Apply dropout on hidden layer activations
        ff_hidden = dropout(ff_hidden, self.dropout_keep_prob, self.train_mode)

        # Feed-forward output projection
        ff_output = tf.layers.dense(
            ff_hidden, self.model_dimension, name="output")

        # Apply dropout on feed-forward output projection
        ff_output = dropout(ff_output, self.dropout_keep_prob, self.train_mode)

        # Add residual connections
        return ff_output + layer_input

    def layer(self, level: int) -> TransformerLayer:
        # Recursive implementation. Outputs of the zeroth layer
        # are normalized inputs.
        if level == 0:
            return TransformerLayer(self.encoder_inputs, self.temporal_mask)

        # Compute the outputs of the previous layer
        prev_layer = self.layer(level - 1)

        with tf.variable_scope("layer_{}".format(level - 1)):
            with tf.variable_scope("self_attention"):
                self_context = self.self_attention_sublayer(prev_layer)

            with tf.variable_scope("feedforward"):
                output_states = self.feedforward_sublayer(self_context)

        # Layer normalization on the encoder outputs
        if self.depth == level:
            output_states = layer_norm(output_states)

        return TransformerLayer(states=output_states, mask=self.temporal_mask)

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return self.layer(self.depth).temporal_states

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return self.input_sequence.temporal_mask

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.train_mode: train}
