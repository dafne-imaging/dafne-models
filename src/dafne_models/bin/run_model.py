#!/bin/env python3
# -*- coding: utf-8 -*-
# tested models for leg: 1610001000 (initial), 1669385545 (final)

import argparse
import numpy as np
import os
import voxel as vx
from dicomUtils import medical_volume_from_path, realign_medical_volume

from ..utils.model_tools import parse_complex_options, ensure_dependencies_from_metadata, \
    ensure_compatible_orientation_inplace, load_model_and_install_deps


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="Path to image/dataset to segment")
    parser.add_argument("other_contrasts", nargs='*', help="Other contrasts to segment")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model file to use")
    parser.add_argument("--classification", nargs=1, metavar='class', type=str, required=False, help="Classification label")
    parser.add_argument("--output", "-o", type=str, required=False, help="Path to save the output", default='.')
    parser.add_argument("--options", "-p", type=str, required=False, help="Options to be passed to the model", default='')
    args = parser.parse_args()

    options=parse_complex_options(args.options)

    try:
        model_classification = args.classification[0].split(',')[0]
    except TypeError:
        model_classification = ''

    print("Loading model")

    model = load_model_and_install_deps(args.model)
    metadata = model.get_metadata()

    image = medical_volume_from_path(args.image_path, reorient_data=False)

    ensure_compatible_orientation_inplace(image, metadata)

    resolution = image.pixel_spacing
    inputs = [image.volume.astype(np.float32)]

    print("Image loaded")

    for contrast in args.other_contrasts:
        contrast_image = medical_volume_from_path(contrast, reorient_data=False)
        contrast_image = realign_medical_volume(contrast_image, image)
        inputs.append(contrast_image.volume.astype(np.float32))

    print("Contrasts loaded")

    dimensionality = model.data_dimensionality
    output_masks = {}
    if dimensionality == 2: # this is a 2D model
        print("2D model")
        n_slices = image.shape[2]
        for i in range(n_slices):
            input_dict = {'image': inputs[0][:, :, i], 'resolution': resolution[:2], 'options': options, 'split_laterality': False, 'classification': model_classification}
            for idx, contrast in enumerate(inputs[1:]):
                input_dict[f'image{idx+1}'] = contrast[:, :, i]
            output = model.apply(input_dict)
            for key, mask in output.items():
                if key not in output_masks:
                    output_masks[key] = np.zeros((image.shape[0], image.shape[1], n_slices), dtype=np.uint8)
                output_masks[key][:, :, i] = mask
    else: # this is a 3D model
        print("3D model")
        input_dict = {'image': inputs[0], 'resolution': resolution, 'options': options, 'split_laterality': False, 'classification': model_classification}
        for idx, contrast in enumerate(inputs[1:]):
            print("Adding contrast", f'image{idx+2}')
            input_dict[f'image{idx+2}'] = contrast
        print("Applying model")
        output = model.apply(input_dict)
        for key, mask in output.items():
            output_masks[key] = mask

    writer = vx.NiftiWriter()

    for key, mask in output_masks.items():
        output_data = vx.MedicalVolume(mask, image.affine)
        if model_orientation is not None and model_orientation != original_orientation:
            output_data.reformat(original_orientation, inplace=True)
        writer.save(output_data, os.path.join(args.output, f'{key}.nii.gz'))


if __name__ == '__main__':
    main()