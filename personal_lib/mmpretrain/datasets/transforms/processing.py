import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
import random
from mmpretrain.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomResizedCropV2(BaseTransform): #Timmのようにrandomにinterpolationを選択
    """Crop the given image to random scale and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        scale (sequence | int): Desired output scale of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str | list[str]): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            'cv2' and 'pillow'. Defaults to 'cv2'.
    """

    def __init__(self,
                 scale: Union[Sequence, int],
                 crop_ratio_range: Tuple[float, float] = (0.08, 1.0),
                 aspect_ratio_range: Tuple[float, float] = (3. / 4., 4. / 3.),
                 max_attempts: int = 10,
                 interpolation: Union[str, List[str]] = ['bilinear', 'bicubic'],
                 backend: str = 'cv2') -> None:
        if isinstance(scale, Sequence):
            assert len(scale) == 2
            assert scale[0] > 0 and scale[1] > 0
            self.scale = scale
        else:
            assert scale > 0
            self.scale = (scale, scale)
        if (crop_ratio_range[0] > crop_ratio_range[1]) or (
                aspect_ratio_range[0] > aspect_ratio_range[1]):
            raise ValueError(
                'range should be of kind (min, max). '
                f'But received crop_ratio_range {crop_ratio_range} '
                f'and aspect_ratio_range {aspect_ratio_range}.')
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        if not isinstance(interpolation, (list, tuple)):
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                     'lanczos')

        self.crop_ratio_range = crop_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    # @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (math.log(self.aspect_ratio_range[0]),
                         math.log(self.aspect_ratio_range[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_w <= w and 0 < target_h <= h:
                offset_h = np.random.randint(0, h - target_h + 1)
                offset_w = np.random.randint(0, w - target_w + 1)

                return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly resized cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = mmcv.imcrop(
            img,
            bboxes=np.array([
                offset_w, offset_h, offset_w + target_w - 1,
                offset_h + target_h - 1
            ]))
        if not isinstance(self.interpolation, (list, tuple)):
            interpolation = self.interpolation
        else:
            interpolation = random.choice(self.interpolation)
        img = mmcv.imresize(
            img,
            tuple(self.scale[::-1]),
            interpolation=interpolation,
            backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(scale={self.scale}'
        repr_str += ', crop_ratio_range='
        repr_str += f'{tuple(round(s, 4) for s in self.crop_ratio_range)}'
        repr_str += ', aspect_ratio_range='
        repr_str += f'{tuple(round(r, 4) for r in self.aspect_ratio_range)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str