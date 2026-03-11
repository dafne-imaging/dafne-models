#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import sys

if 'generate_convert' not in locals() and 'generate_convert' not in globals():
    from dafne_models.common import generate_convert

try:
    from dafne_dl import DynamicTorchModel
except ModuleNotFoundError:
    from dl import DynamicTorchModel


def init_unetr():
    from monai.networks.nets import UNETR
    model = UNETR(
        in_channels=1,
        out_channels=3,  # BACKGROUND, RIGHT KIDNEY (left on image), LEFT KIDNEY (right on image)
        img_size=(80, 80, 80),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
    return model


def unetr_apply(modelObj: DynamicTorchModel, data: dict):
    from dafne_dl.interfaces import WrongDimensionalityError
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference

    DEFAULT_OVERLAP = 0.2

    input_array = data['image'].astype(np.float32)
    overlap = data.get('overlap', DEFAULT_OVERLAP)

    if len(input_array.shape) != 3:
        raise WrongDimensionalityError("Input image must be 3D")

    def largest_cluster(array: np.ndarray) -> np.ndarray:
        """Given a mask array, return a new mask array containing only the largesr cluster.

        Args:
            array (np.ndarray): mask array with values 1 (inside) or 0 (outside)

        Returns:
            np.ndarray: mask array with only a single connect cluster of pixels.
        """
        import scipy.ndimage as ndi
        # Label all features in the array
        label_img, cnt = ndi.label(array)
        # Find the label of the largest feature
        labels = range(1, cnt + 1)
        size = [np.count_nonzero(label_img == l) for l in labels]
        max_label = labels[size.index(np.amax(size))]
        # Return a mask corresponding to the largest feature
        return label_img == max_label

    # Normalize data
    input_array = (input_array - np.average(input_array)) / np.std(input_array)

    # Convert to NCHW[D] format: (1,1,y,x,z)
    # NCHW[D] stands for: batch N, channels C, height H, width W, depth D
    input_array = input_array.transpose(1, 0, 2)  # from (x,y,z) to (y,x,z)
    input_array = np.expand_dims(input_array, axis=(0, 1))

    # Convert to tensor
    input_tensor = torch.tensor(input_array, device=modelObj.device)

    # Load model weights
    model = modelObj.model
    model.eval()

    # Calculate model output (decrease overlap parameter for faster but less accurate results)
    with torch.no_grad():
        output_tensor = sliding_window_inference(input_tensor, (80, 80, 80), 4, model, overlap=overlap,
                                                 device=modelObj.device, progress=True)

    # From probabilities for each channel to label image
    output_tensor = torch.argmax(output_tensor, dim=1)

    # Convert to numpy
    output_array = output_tensor.numpy(force=True)[0, :, :, :]

    # Transpose to original shape
    output_array = output_array.transpose(1, 0, 2)  # from (y,x,z) to (x,y,z)

    left_kidney = largest_cluster(output_array == 2)
    right_kidney = largest_cluster(output_array == 1)
    return {
        'Left kidney': left_kidney.astype(np.uint8),
        'Right kidney': right_kidney.astype(np.uint8)
    }


generate_convert(model_id='f3749819-ab5a-4769-bcd1-9b2885fb6c14',
                 default_weights_path=os.path.join('weights', 'UNETR_kidneys_v1.pth'),
                 model_name_prefix='Kidney_3D_UNETR',
                 model_create_function=init_unetr,
                 model_apply_function=unetr_apply,
                 model_learn_function=None,
                 dimensionality=3,
                 model_type=DynamicTorchModel
                 )
