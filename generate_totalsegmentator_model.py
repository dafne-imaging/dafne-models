#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Dafne .model plugins wrapping TotalSegmentator.

Produces:
  TotalSegmentatorCT_<timestamp>.model   — 117-class CT segmentation
  TotalSegmentatorMR_<timestamp>.model   — 50-class MR segmentation

Run:
  python generate_totalsegmentator_model.py [output_dir]

TotalSegmentator weights are stored under Dafne's own data directory
(via appdirs) rather than ~/.totalsegmentator.
"""

import json
import os
import sys

from dafne_dl.DynamicDummyModel import DynamicDummyModel

MODEL_ID_CT = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
MODEL_ID_MR = 'b2c3d4e5-f6a7-8901-bcde-f12345678901'

TIMESTAMP = 1710001000  # fixed so the filename is stable across re-runs


# ---------------------------------------------------------------------------
# CT apply function
# All imports and data must be local — source is serialized and re-executed.
# ---------------------------------------------------------------------------

def apply_totalsegmentator_ct(modelObj, data):
    import os
    import numpy as np
    import nibabel as nib
    from appdirs import AppDirs
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map

    MIN_LABEL_VOXELS = 5  # drop masks with fewer than this many positive voxels (organ not present)
    CT_VARIANT_SUBSETS = {
        '': None,
        'Abdomen': [
            'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver',
            'stomach', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left',
            'small_bowel', 'duodenum', 'colon', 'urinary_bladder', 'prostate',
            'kidney_cyst_left', 'kidney_cyst_right',
            'portal_vein_and_splenic_vein', 'inferior_vena_cava',
            'iliac_artery_left', 'iliac_artery_right',
            'iliac_vena_left', 'iliac_vena_right',
        ],
        'Thorax': [
            'lung_upper_lobe_left', 'lung_lower_lobe_left',
            'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right',
            'trachea', 'esophagus', 'heart',
            'aorta', 'pulmonary_vein', 'superior_vena_cava', 'inferior_vena_cava',
            'brachiocephalic_trunk', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right',
            'subclavian_artery_right', 'subclavian_artery_left',
            'common_carotid_artery_right', 'common_carotid_artery_left',
            'atrial_appendage_left', 'sternum', 'costal_cartilages',
        ],
        'Spine': [
            'spinal_cord', 'sacrum', 'vertebrae_S1',
            'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1',
            'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8',
            'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3',
            'vertebrae_T2', 'vertebrae_T1',
            'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4',
            'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1',
        ],
        'Pelvis': [
            'sacrum', 'hip_left', 'hip_right', 'femur_left', 'femur_right',
            'urinary_bladder', 'prostate',
            'iliac_artery_left', 'iliac_artery_right',
            'iliac_vena_left', 'iliac_vena_right',
            'gluteus_maximus_left', 'gluteus_maximus_right',
            'gluteus_medius_left', 'gluteus_medius_right',
            'gluteus_minimus_left', 'gluteus_minimus_right',
            'iliopsoas_left', 'iliopsoas_right',
        ],
        'Head & Neck': [
            'brain', 'skull', 'thyroid_gland', 'trachea',
            'common_carotid_artery_right', 'common_carotid_artery_left',
            'brachiocephalic_trunk',
            'subclavian_artery_right', 'subclavian_artery_left',
        ],
        'Bones': [
            'skull', 'sacrum', 'vertebrae_S1',
            'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1',
            'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8',
            'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3',
            'vertebrae_T2', 'vertebrae_T1',
            'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4',
            'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1',
            'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5',
            'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10',
            'rib_left_11', 'rib_left_12',
            'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5',
            'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10',
            'rib_right_11', 'rib_right_12',
            'sternum', 'costal_cartilages',
            'humerus_left', 'humerus_right',
            'scapula_left', 'scapula_right',
            'clavicula_left', 'clavicula_right',
            'femur_left', 'femur_right',
            'hip_left', 'hip_right',
        ],
        'Muscles': [
            'gluteus_maximus_left', 'gluteus_maximus_right',
            'gluteus_medius_left', 'gluteus_medius_right',
            'gluteus_minimus_left', 'gluteus_minimus_right',
            'autochthon_left', 'autochthon_right',
            'iliopsoas_left', 'iliopsoas_right',
        ],
    }

    app_dirs = AppDirs('Dafne', 'Dafne-imaging')
    weights_dir = os.path.join(app_dirs.user_data_dir, 'totalsegmentator_weights')
    os.makedirs(weights_dir, exist_ok=True)
    os.environ['TOTALSEG_WEIGHTS_PATH'] = weights_dir

    parts = data['classification'].split(',')
    variant = parts[1].strip() if len(parts) > 1 else ''
    roi_subset = CT_VARIANT_SUBSETS.get(variant, None)

    nifti_img = nib.Nifti1Image(data['image'], data['affine'])
    device = 'gpu' if modelObj.device.type != 'cpu' else 'cpu'
    seg = totalsegmentator(nifti_img, output=None, ml=True,
                           roi_subset=roi_subset, task='total', device=device)

    seg_data = seg.get_fdata().astype(np.uint8)
    label_map = class_map['total']
    masks = {
        name: (seg_data == lid).astype(np.uint8)
        for lid, name in label_map.items()
        if roi_subset is None or name in roi_subset
    }
    return {name: mask for name, mask in masks.items() if mask.sum() >= MIN_LABEL_VOXELS}


# ---------------------------------------------------------------------------
# MR apply function
# ---------------------------------------------------------------------------

def apply_totalsegmentator_mr(modelObj, data):
    import os
    import numpy as np
    import nibabel as nib
    from appdirs import AppDirs
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map

    MIN_LABEL_VOXELS = 5  # drop masks with fewer than this many positive voxels (organ not present)
    MR_VARIANT_SUBSETS = {
        '': None,
        'Abdomen': [
            'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver',
            'stomach', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left',
            'small_bowel', 'duodenum', 'colon', 'urinary_bladder', 'prostate',
            'portal_vein_and_splenic_vein', 'inferior_vena_cava',
            'iliac_artery_left', 'iliac_artery_right',
            'iliac_vena_left', 'iliac_vena_right',
        ],
        'Thorax': [
            'lung_left', 'lung_right', 'esophagus', 'heart',
            'aorta', 'inferior_vena_cava',
        ],
        'Spine': [
            'sacrum', 'vertebrae', 'intervertebral_discs', 'spinal_cord',
        ],
        'Pelvis': [
            'sacrum', 'hip_left', 'hip_right', 'femur_left', 'femur_right',
            'urinary_bladder', 'prostate',
            'iliac_artery_left', 'iliac_artery_right',
            'iliac_vena_left', 'iliac_vena_right',
            'gluteus_maximus_left', 'gluteus_maximus_right',
            'gluteus_medius_left', 'gluteus_medius_right',
            'gluteus_minimus_left', 'gluteus_minimus_right',
            'iliopsoas_left', 'iliopsoas_right',
        ],
        'Bones': [
            'sacrum', 'vertebrae',
            'humerus_left', 'humerus_right',
            'scapula_left', 'scapula_right',
            'clavicula_left', 'clavicula_right',
            'femur_left', 'femur_right',
            'hip_left', 'hip_right',
        ],
        'Muscles': [
            'gluteus_maximus_left', 'gluteus_maximus_right',
            'gluteus_medius_left', 'gluteus_medius_right',
            'gluteus_minimus_left', 'gluteus_minimus_right',
            'autochthon_left', 'autochthon_right',
            'iliopsoas_left', 'iliopsoas_right',
        ],
    }

    app_dirs = AppDirs('Dafne', 'Dafne-imaging')
    weights_dir = os.path.join(app_dirs.user_data_dir, 'totalsegmentator_weights')
    os.makedirs(weights_dir, exist_ok=True)
    os.environ['TOTALSEG_WEIGHTS_PATH'] = weights_dir

    parts = data['classification'].split(',')
    variant = parts[1].strip() if len(parts) > 1 else ''
    roi_subset = MR_VARIANT_SUBSETS.get(variant, None)

    nifti_img = nib.Nifti1Image(data['image'], data['affine'])
    device = 'gpu' if modelObj.device.type != 'cpu' else 'cpu'
    seg = totalsegmentator(nifti_img, output=None, ml=True,
                           roi_subset=roi_subset, task='total_mr', device=device)

    seg_data = seg.get_fdata().astype(np.uint8)
    label_map = class_map['total_mr']
    masks = {
        name: (seg_data == lid).astype(np.uint8)
        for lid, name in label_map.items()
        if roi_subset is None or name in roi_subset
    }
    return {name: mask for name, mask in masks.items() if mask.sum() >= MIN_LABEL_VOXELS}


# ---------------------------------------------------------------------------
# Save helpers
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

    ct_variants = ['', 'Abdomen', 'Thorax', 'Spine', 'Pelvis', 'Head & Neck', 'Bones', 'Muscles']
    mr_variants = ['', 'Abdomen', 'Thorax', 'Spine', 'Pelvis', 'Bones', 'Muscles']

    ct_model = DynamicDummyModel(
        model_id=MODEL_ID_CT,
        apply_model_function=apply_totalsegmentator_ct,
        timestamp_id=TIMESTAMP,
        data_dimensionality=3,
        metadata={
            'model_name': 'TotalSegmentator CT',
            'model_type': 'DynamicDummyModel',
            'dimensionality': '3',
            'variants': ct_variants,
            'categories': [['CT', 'TotalSegmentator']],
            'orientation': '',
            'info': {
                'Description': 'Segments 117 anatomical structures in CT using TotalSegmentator v2',
                'Author': 'Wasserthal et al.',
                'Modality': 'CT',
                'Reference': 'Wasserthal et al., Radiology: AI 2023',
            },
            'dependencies': {
                'totalsegmentator': 'TotalSegmentator >= 2.0  --SimpleITK ++!SimpleITK-SimpleElastix',
                'appdirs': 'appdirs',
            },
        },
    )
    save_model(ct_model, 'TotalSegmentatorCT', output_dir)

    mr_model = DynamicDummyModel(
        model_id=MODEL_ID_MR,
        apply_model_function=apply_totalsegmentator_mr,
        timestamp_id=TIMESTAMP,
        data_dimensionality=3,
        metadata={
            'model_name': 'TotalSegmentator MR',
            'model_type': 'DynamicDummyModel',
            'dimensionality': '3',
            'variants': mr_variants,
            'categories': [['MRI', 'TotalSegmentator']],
            'orientation': '',
            'info': {
                'Description': 'Segments 50 anatomical structures in MRI using TotalSegmentator v2',
                'Author': 'Wasserthal et al.',
                'Modality': 'MRI',
                'Reference': 'Wasserthal et al., Radiology 2025',
            },
            'dependencies': {
                'totalsegmentator': 'TotalSegmentator >= 2.0 --SimpleITK ++!SimpleITK-SimpleElastix',
                'appdirs': 'appdirs',
            },
        },
    )
    save_model(mr_model, 'TotalSegmentatorMR', output_dir)
