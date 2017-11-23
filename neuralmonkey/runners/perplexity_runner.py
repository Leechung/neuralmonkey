from typing import Dict, List, Set, cast

from typeguard import check_argument_types
import tensorflow as tf
import numpy as np

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)


class PerplexityExecutable(Executable):
    def __init__(self, all_coders: Set[ModelPart],
                 xent_op: tf.Tensor,
                 num_sessions: int) -> None:
        self._all_coders = all_coders
        self._xent_op = xent_op
        self._num_sessions = num_sessions
        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return (self._all_coders,
                {"xents": self._xent_op},
                [{} for _ in range(self._num_sessions)])

    def collect_results(self, results: List[Dict]) -> None:
        perplexities = np.mean([2 ** res["xents"] for res in results], axis=0)
        xent = float(np.mean([res["xents"] for res in results]))
        self.result = ExecutionResult(
            outputs=perplexities.tolist(),
            losses=[xent],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class PerplexityRunner(BaseRunner):
    def __init__(self,
                 output_series: str,
                 decoder: Decoder) -> None:
        check_argument_types()
        BaseRunner.__init__(self, output_series, decoder)

        self._decoder_xent = cast(Decoder, self._decoder).train_xents

    def get_executable(self, compute_losses=False,
                       summaries=True,
                       num_sessions=1) -> PerplexityExecutable:
        return PerplexityExecutable(self.all_coders,
                                    self._decoder_xent,
                                    num_sessions)

    @property
    def loss_names(self) -> List[str]:
        return ["xent"]
