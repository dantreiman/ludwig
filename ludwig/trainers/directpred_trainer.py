#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
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
# ==============================================================================
"""This module contains the class and auxiliary methods of a model."""
import gc
import logging
import math
import os
import os.path
import signal
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

from ludwig.constants import COMBINED, DEFAULT_BATCH_SIZE, LOSS, MODEL_ECD, TEST, TRAINING, VALIDATION
from ludwig.data.dataset.base import Dataset
from ludwig.globals import (
    is_progressbar_disabled,
    MODEL_HYPERPARAMETERS_FILE_NAME,
    TRAINING_CHECKPOINTS_DIR_PATH,
    TRAINING_PROGRESS_TRACKER_FILE_NAME,
)
from ludwig.models.ecd import ECD
from ludwig.models.predictor import Predictor
from ludwig.modules.metric_modules import get_improved_fun, get_initial_validation_value
from ludwig.modules.optimization_modules import create_clipper, create_optimizer
from ludwig.progress_bar import LudwigProgressBar
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.trainers.registry import register_trainer
from ludwig.trainers.trainer import Trainer
from ludwig.utils import time_utils
from ludwig.utils.checkpoint_utils import Checkpoint, CheckpointManager
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.horovod_utils import return_first
from ludwig.utils.math_utils import exponential_decay, learning_rate_warmup, learning_rate_warmup_distributed
from ludwig.utils.metric_utils import get_metric_names, TrainerMetric
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.torch_utils import get_torch_device, reg_loss
from ludwig.utils.trainer_utils import (
    append_metrics,
    get_final_steps_per_checkpoint,
    get_new_progress_tracker,
    get_total_steps,
    ProgressTracker,
)

logger = logging.getLogger(__name__)


@register_trainer(MODEL_ECD, default=True)
class DirectPredTrainer(Trainer):
    """DirectPredTrainer trains a self-supervised model by tabular data augmentation."""

    @staticmethod
    def get_schema_cls():
        return ECDTrainerConfig

    def __init__(
        self,
        config: ECDTrainerConfig,
        model: ECD,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        report_tqdm_to_ray=False,
        random_seed: float = default_random_seed,
        horovod: Optional[Dict] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Trains a model with a set of options and hyperparameters listed below. Customizable.

        :param model: Underlying Ludwig model
        :type model: `ludwig.models.ecd.ECD`
        :param resume: Resume training a model that was being trained. (default: False).
        :type resume: Boolean
        :param skip_save_model: Disables saving model weights and hyperparameters each time the model improves. By
                default Ludwig saves model weights after each round of evaluation the validation metric (improves, but
                if the model is really big that can be time consuming. If you do not want to keep the weights and just
                find out what performance a model can get with a set of hyperparameters, use this parameter to skip it,
                but the model will not be loadable later on. (default: False).
        :type skip_save_model: Boolean
        :param skip_save_progress: Disables saving progress each round of evaluation. By default Ludwig saves weights
                and stats after each round of evaluation for enabling resuming of training, but if the model is really
                big that can be time consuming and will uses twice as much space, use this parameter to skip it, but
                training cannot be resumed later on. (default: False).
        :type skip_save_progress: Boolean
        :param skip_save_log: Disables saving TensorBoard logs. By default Ludwig saves logs for the TensorBoard, but if
                it is not needed turning it off can slightly increase the overall speed. (default: False).
        :type skip_save_log: Boolean
        :param callbacks: List of `ludwig.callbacks.Callback` objects that provide hooks into the Ludwig pipeline.
                (default: None).
        :type callbacks: list
        :param report_tqdm_to_ray: Enables using the ray based tqdm Callback for progress bar reporting
        :param random_seed: Default initialization for the random seeds (default: 42).
        :type random_seed: Float
        :param horovod: Horovod parameters (default: None).
        :type horovod: dict
        :param device: Device to load the model on from a saved checkpoint (default: None).
        :type device: str
        :param config: `ludwig.schema.trainer.BaseTrainerConfig` instance that specifies training hyperparameters
                (default: `ludwig.schema.trainer.ECDTrainerConfig()`).
        """
        super().__init__(
            config,
            model,
            resume,
            skip_save_model,
            skip_save_progress,
            skip_save_log,
            callbacks,
            report_tqdm_to_ray,
            random_seed,
            horovod,
            device,
            **kwargs,
        )
        self.bootstrapping = True
        self.augment_rate = 0.2

    def train_step(
        self, inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Performs a single training step.

        Params:
            inputs: A dictionary of input data, from feature name to tensor.
            targets: A dictionary of target data, from feature name to tensor.

        Returns:
            A tuple of the loss tensor and a dictionary of loss for every output feature.
        """
        if not self.bootstrapping:
            return super().train_step(inputs, targets)
        if isinstance(self.optimizer, torch.optim.LBFGS):
            assert (False, "Make this work for LBFGS optimizer (maybe)")

        self.optimizer.zero_grad()

        # Create augmented batch
        first_key = list(inputs.keys())[0]
        N = inputs[first_key].shape[0]  # Batch size
        n_2 = N // 2
        X = {k: v[:n_2] for k, v in inputs.items()}
        Y = {k: v[:n_2] for k, v in targets.items()}
        Z = {k: v[n_2:] for k, v in inputs.items()}  # Random train examples not used
        masks = {
            k: (
                torch.rand((n_2, 1), device=self.model.device)
                if len(X[k].shape) == 2
                else torch.rand((n_2,), device=self.model.device)
            )
            > self.augment_rate
            for k in inputs.keys()
        }
        # Augmented copy of x formed by sampling from empirical distribution
        X_hat = {k: masks[k] * x + (~masks[k]) * Z[k] for k, x in X.items()}

        x_outputs = self.model((X, Y))
        x_hat_outputs = self.model((X_hat, Y))

        def get_hidden(d):
            hidden_keys = [k for k in d.keys() if k.endswith("::last_hidden")]
            return d[hidden_keys[0]]

        def get_embedding(d):
            projection_keys = [k for k in d.keys() if k.endswith("::projection_input")]
            return d[projection_keys[0]]

        # Predictor output = "average_product_rating_quantized::last_hidden"
        # Embeddings = "average_product_rating_quantized::projection_input"

        predictor_output = get_hidden(x_outputs)
        x_hat_embedding = get_embedding(x_hat_outputs).detach()

        # Compute bootstrap loss
        predictor_loss = torch.nn.functional.mse_loss(predictor_output, x_hat_embedding.detach())

        # Add regularization loss
        if self.regularization_type is not None and self.regularization_lambda != 0:
            decay_loss = reg_loss(
                self.model, self.regularization_type, l1=self.regularization_lambda, l2=self.regularization_lambda
            )

        loss = predictor_loss + decay_loss
        all_losses = {"loss": loss, "predictor loss": predictor_loss, "l1 loss": decay_loss}
        # loss, all_losses = self.model.train_loss(
        #     targets, model_outputs, self.regularization_type, self.regularization_lambda
        # )

        # Begin the backward pass
        variables = self.model.parameters()
        loss.backward()

        if self.horovod:
            # Wait for gradient aggregation to complete before clipping the gradients
            self.optimizer.synchronize()

        # Clip gradients
        self.clip_grads(variables)

        # Apply gradient updates
        if self.horovod:
            # Because we already synchronized above, we can doing so here
            with self.optimizer.skip_synchronize():
                self.optimizer.step()
        else:
            self.optimizer.step()

        return loss, all_losses
