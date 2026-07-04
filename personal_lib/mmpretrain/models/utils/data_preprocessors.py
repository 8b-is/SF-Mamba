import math
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmpretrain.registry import MODELS
from mmpretrain.models.utils import ClsDataPreprocessor


@MODELS.register_module()
class DistilClsDataPreprocessor(ClsDataPreprocessor):
    """Image pre-processor for classification tasks with distillation.

    Comparing with the :class:`mmengine.model.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        to_onehot (bool): Whether to generate one-hot format gt-labels and set
            to data samples. Defaults to False.
        num_classes (int, optional): The number of classes. Defaults to None.
        batch_augments (dict, optional): The batch augmentations settings,
            including "augments" and "probs". For more details, see
            :class:`mmpretrain.models.RandomBatchAugment`.
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """

        if "processed_data" in data.keys(): # Teacher preprocessor already create batch. Use the same batch aug.
            return data["processed_data"]
        else:
            return super().forward(data, training)
        