#!/bin/env python3
# -*- coding: utf-8 -*-
# tested models for leg: 1610001000 (initial), 1669385545 (final)

import argparse
import numpy as np
import os
import voxel as vx
from dicomUtils import medical_volume_from_path, realign_medical_volume
from dafne_dl.model_loaders import generic_load_model
import flexidep
import ast
import pprint
import dill

APP_STRING = 'network.dafne_models.run_model'

def parse_complex_options(options_string):
    def parse_value(value):
        value = value.strip()
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    options = {}
    current_key = None
    current_value = ''
    stack = []
    in_quotes = False
    quote_char = None

    for char in options_string:
        if char in ('"', "'") and (not in_quotes or char == quote_char):
            in_quotes = not in_quotes
            quote_char = char if in_quotes else None
            current_value += char
        elif char == ',' and not stack and not in_quotes:
            if current_key:
                options[current_key] = parse_value(current_value)
                current_key = None
                current_value = ''
        elif char == '=' and not stack and not in_quotes:
            current_key = current_value.strip()
            current_value = ''
        elif char == '[' and not in_quotes:
            stack.append('[')
            current_value += char
        elif char == ']' and not in_quotes:
            if stack and stack[-1] == '[':
                stack.pop()
            current_value += char
        else:
            current_value += char

    if current_key:
        options[current_key] = parse_value(current_value)

    return options

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

    with open(args.model, 'rb') as f:
        input_dict = dill.load(f)

    metadata = input_dict.get('metadata', {})
    print("Model metadata:")
    pprint.pp(metadata)

    # check and install model dependencies
    dependencies = metadata.get('dependencies', {})

    dependency_manager = flexidep.DependencyManager(
        config_file=None,
        config_string=None,
        unique_id=APP_STRING,
        interactive_initialization=False,
        use_gui=False,
        install_local=False,
        package_manager=flexidep.PackageManagers.pip,
        extra_command_line='',
    )

    for package, alternative_str in dependencies.items():
        print("Processing package", package)
        dependency_manager.process_single_package(package, alternative_str)

    # build the model object. Now that the dependencies are installed, the model can be loaded
    model = generic_load_model(input_dict)

    # check if the model has a specific orientation
    model_orientation = metadata.get('orientation', None)

    if isinstance(model_orientation, str):
        # the orientation is a string (Axial/Transversal, Sagittal, Coronal)
        model_orientation = model_orientation.lower()
        if model_orientation.startswith('a') or model_orientation.startswith('t'):
            model_orientation = ('LR', 'AP', 'SI')
        elif model_orientation.startswith('s'):
            model_orientation = ('AP', 'IS', 'LR')
        elif model_orientation.startswith('c'):
            model_orientation = ('LR', 'SI', 'AP')
        else:
            print("Unknown orientation")
            model_orientation = None

    image = medical_volume_from_path(args.image_path, reorient_data=False)

    original_orientation = image.orientation

    if model_orientation is not None and model_orientation != original_orientation:
        image.reformat(model_orientation, inplace=True)

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