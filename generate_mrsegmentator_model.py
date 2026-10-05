#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a Dafne .model plugin wrapping MRSegmentator.

Produces:
  MRSegmentator_<timestamp>.model   — 40-class MRI/CT segmentation,
                                       optionally + 10 body-composition classes

Run:
  python generate_mrsegmentator_model.py [output_dir]

MRSegmentator weights are stored under Dafne's own data directory (via
appdirs) rather than ~/.mrsegmentator.

See MRSegmentator_Integration_Plan.md for the full design rationale.
"""

import json
import os
import sys

from dafne_dl.DynamicDummyModel import DynamicDummyModel

MODEL_ID = 'c3d4e5f6-a7b8-9012-cdef-234567890123'

TIMESTAMP = 1720000000  # fixed so the filename is stable across re-runs

# Base model classes (ids 1-40), from MRSegmentator's README class table.
BASE_LABELS = {
    1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gallbladder',
    5: 'liver', 6: 'stomach', 7: 'pancreas', 8: 'right_adrenal_gland',
    9: 'left_adrenal_gland', 10: 'left_lung', 11: 'right_lung', 12: 'heart',
    13: 'aorta', 14: 'inferior_vena_cava', 15: 'portal_vein_and_splenic_vein',
    16: 'left_iliac_artery', 17: 'right_iliac_artery', 18: 'left_iliac_vena',
    19: 'right_iliac_vena', 20: 'esophagus', 21: 'small_bowel', 22: 'duodenum',
    23: 'colon', 24: 'urinary_bladder', 25: 'spine', 26: 'sacrum',
    27: 'left_hip', 28: 'right_hip', 29: 'left_femur', 30: 'right_femur',
    31: 'left_autochthonous_muscle', 32: 'right_autochthonous_muscle',
    33: 'left_iliopsoas_muscle', 34: 'right_iliopsoas_muscle',
    35: 'left_gluteus_maximus', 36: 'right_gluteus_maximus',
    37: 'left_gluteus_medius', 38: 'right_gluteus_medius',
    39: 'left_gluteus_minimus', 40: 'right_gluteus_minimus',
}

# Body-composition model classes (ids 1-10), MRI only.
BODY_COMP_LABELS = {
    1: 'subcutaneous_fat', 2: 'visceral_fat', 3: 'left_rectus_abdominis',
    4: 'right_rectus_abdominis', 5: 'left_oblique_muscle',
    6: 'right_oblique_muscle', 7: 'left_quadratus_lumborum',
    8: 'right_quadratus_lumborum', 9: 'abdominal_subcutaneous_fat',
    10: 'gluteofemoral_fat',
}


# ---------------------------------------------------------------------------
# Apply function
# All imports and data must be local — source is serialized and re-executed.
# ---------------------------------------------------------------------------

def apply_mrsegmentator(modelObj, data):
    import os
    import tempfile

    import nibabel as nib
    import numpy as np
    from appdirs import AppDirs
    from mrsegmentator import inference

    MIN_LABEL_VOXELS = 5  # drop masks with fewer than this many positive voxels (organ not present)

    BASE_LABELS = {
        1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gallbladder',
        5: 'liver', 6: 'stomach', 7: 'pancreas', 8: 'right_adrenal_gland',
        9: 'left_adrenal_gland', 10: 'left_lung', 11: 'right_lung', 12: 'heart',
        13: 'aorta', 14: 'inferior_vena_cava', 15: 'portal_vein_and_splenic_vein',
        16: 'left_iliac_artery', 17: 'right_iliac_artery', 18: 'left_iliac_vena',
        19: 'right_iliac_vena', 20: 'esophagus', 21: 'small_bowel', 22: 'duodenum',
        23: 'colon', 24: 'urinary_bladder', 25: 'spine', 26: 'sacrum',
        27: 'left_hip', 28: 'right_hip', 29: 'left_femur', 30: 'right_femur',
        31: 'left_autochthonous_muscle', 32: 'right_autochthonous_muscle',
        33: 'left_iliopsoas_muscle', 34: 'right_iliopsoas_muscle',
        35: 'left_gluteus_maximus', 36: 'right_gluteus_maximus',
        37: 'left_gluteus_medius', 38: 'right_gluteus_medius',
        39: 'left_gluteus_minimus', 40: 'right_gluteus_minimus',
    }
    BODY_COMP_LABELS = {
        1: 'subcutaneous_fat', 2: 'visceral_fat', 3: 'left_rectus_abdominis',
        4: 'right_rectus_abdominis', 5: 'left_oblique_muscle',
        6: 'right_oblique_muscle', 7: 'left_quadratus_lumborum',
        8: 'right_quadratus_lumborum', 9: 'abdominal_subcutaneous_fat',
        10: 'gluteofemoral_fat',
    }

    app_dirs = AppDirs('Dafne', 'Dafne-imaging')
    weights_dir = os.path.join(app_dirs.user_data_dir, 'mrsegmentator_weights')
    os.makedirs(weights_dir, exist_ok=True)
    os.environ['MRSEG_WEIGHTS_PATH'] = weights_dir

    parts = data['classification'].split(',')
    variant = parts[1].strip() if len(parts) > 1 else ''
    cpu_only = modelObj.device.type == 'cpu'

    def run_pass(model_name, label_map):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, 'input.nii.gz')
            out_dir = os.path.join(tmpdir, 'out')
            nib.save(nib.Nifti1Image(data['image'], data['affine']), in_path)

            inference.infer([in_path], out_dir, cpu_only=cpu_only, model_name=model_name)

            seg_img = nib.load(os.path.join(out_dir, 'input_seg.nii.gz'))
            seg_data = np.asanyarray(seg_img.dataobj).astype(np.uint8)

        return {
            name: (seg_data == lid).astype(np.uint8)
            for lid, name in label_map.items()
        }

    masks = run_pass('base', BASE_LABELS)
    if variant == 'BodyComposition':
        masks.update(run_pass('body_comp', BODY_COMP_LABELS))

    return {name: mask for name, mask in masks.items() if mask.sum() >= MIN_LABEL_VOXELS}


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_model(model, name_prefix, output_dir):
    filename = os.path.join(output_dir, f'{name_prefix}_{model.timestamp_id}.model')
    with open(filename, 'wb') as f:
        model.dump(f)
    print(f'Saved {filename}')

    json_path = os.path.join(output_dir, f'{name_prefix}.json')
    with open(json_path, 'w') as f:
        json.dump(model.get_metadata(), f, indent=4)
    print(f'Saved {json_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'models'
    os.makedirs(output_dir, exist_ok=True)

    model = DynamicDummyModel(
        model_id=MODEL_ID,
        apply_model_function=apply_mrsegmentator,
        timestamp_id=TIMESTAMP,
        data_dimensionality=3,
        metadata={
            'model_name': 'MRSegmentator',
            'model_type': 'DynamicDummyModel',
            'dimensionality': '3',
            'variants': ['', 'BodyComposition'],
            'categories': [['MRI', 'MRSegmentator']],
            'orientation': '',
            'info': {
                'Description': 'Segments 40 organs/structures in MRI (abdomen/pelvis/thorax), '
                                'also works on CT. "BodyComposition" variant adds 10 further '
                                'MRI-only body-composition classes.',
                'Author': 'Haentze et al.',
                'Modality': 'MRI',
                'Reference': 'https://doi.org/10.1148/ryai.240777 '
                             '(body composition: https://doi.org/10.1101/2025.06.03.25328867)',
            },
            'dependencies': {
                # nnunetv2 >= 2.6.2 ships an experimental "Primus" trainer
                # (nnunetv2/training/nnUNetTrainer/primus/primus_trainers.py;
                # confirmed present in 2.6.2/2.7.0/2.8.0, absent in 2.6.0)
                # that gets imported unconditionally during trainer
                # auto-discovery, regardless of which trainer the loaded
                # checkpoint actually needs. Its import chain (timm ->
                # torchvision) raises "RuntimeError: operator
                # torchvision::nms does not exist" in a plain CPU/torch
                # install, and nnunetv2's recursive class finder only
                # catches ModuleNotFoundError, not RuntimeError, so this
                # aborts model loading entirely. mrsegmentator's own pin
                # (nnunetv2>=2.2.1,<=2.8.0) still allows an affected
                # version, so pin below that here.
                #
                # Must be processed BEFORE 'mrsegmentator' (dict order =
                # install order): flexidep's process_single_package() skips
                # a dependency outright if the module already imports
                # (regardless of version — it never checks the version spec
                # once something is importable), so if 'mrsegmentator' ran
                # first and pulled in nnunetv2==2.8.0 transitively, this
                # entry would be silently skipped and the pin would never
                # take effect. Installing nnunetv2 first means pip installs
                # our pinned version for real, and mrsegmentator's later
                # install (no --upgrade) leaves an already-satisfying
                # nnunetv2 alone.
                'nnunetv2': 'nnunetv2 <= 2.6.0',
                'mrsegmentator': 'mrsegmentator >= 2.0',
                'nibabel': 'nibabel',
                'appdirs': 'appdirs',
            },
        },
    )
    save_model(model, 'MRSegmentator', output_dir)
