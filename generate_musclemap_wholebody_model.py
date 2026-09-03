#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2024 Dafne-Imaging Team
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
"""
Dafne plugin reimplementing the MuscleMap whole-body muscle segmentation model
(https://github.com/MuscleMap/MuscleMap) as a native Dafne DynamicTorchModel.

NOTE: whole-body is the ONLY MuscleMap region model this repo targets. The
other region-specific models (abdomen, forearm, leg, pelvis, thigh) are
frozen/unmaintained upstream -- do not add plugins for them, and treat
generate_musclemap_thigh_model.py (if still present) as superseded/retired.

This does NOT call out to the MuscleMap package at runtime. It reimplements
the underlying network (a plain MONAI UNet, spatial_dims=2 -- MuscleMap runs
it slice-by-slice over 3D volumes via monai.inferers.SliceInferer) and a
from-scratch pre/postprocessing pipeline that mirrors MuscleMap's own
mm_segment.py pipeline as closely as possible for a single 2D slice:

  pre_transforms:
    Spacingd(pixdim, mode="bilinear")       -- resample in-plane to 1x1 mm
    NormalizeIntensityd(nonzero=True)       -- zero-mean/unit-std over nonzero voxels
    CropForegroundd(margin=20)              -- crop to foreground bbox (computed
                                                on the *normalized* image, i.e.
                                                after the step above)
    SpatialPadd(spatial_size=(256,256,1))   -- pad up to a hardcoded 256x256
                                                floor, regardless of the
                                                region's sliding-window roi_size
  inference:
    SliceInferer/sliding_window_inference(roi_size, overlap=0.9, mode="gaussian")
  post_transforms:
    Invertd(nearest_interp=False)           -- resample the RAW per-class
                                                logits back to native
                                                resolution with LINEAR
                                                interpolation, BEFORE argmax
    AsDiscreted(argmax=True)                -- discretize only after inversion
    RemapLabels                             -- consecutive channel index ->
                                                sparse label code (folded here
                                                into LABEL_MAP directly)
  final postprocessing (run_inference -> connected_chunks):
    keep only the largest connected component per label

Getting the Invertd-before-argmax ordering right matters a lot: discretizing
at (downsampled) model resolution and then nearest-neighbor-upsampling the
label map (an earlier version of this plugin did that) produces visibly worse
boundaries than resampling the soft per-class output with linear
interpolation and discretizing at native resolution, which is what MuscleMap
actually does.

Uses MuscleMap v1.4 (the current default as of this writing): 113 muscle/bone
labels, out_channels=114. The 113 foreground output channels correspond 1:1
to the sorted list of unique label codes used by MuscleMap (see
contrast_agnostic_wholebody_model.json), exactly mirroring the RemapLabels
post-processing step MuscleMap applies only for the wholebody model.

The pretrained weights (MIT-licensed, Zenodo record 21929873, model v1.4,
"MuscleMap Toolbox -- Contrast-Agnostic Wholebody Muscle Segmentation Model
Weights") are bundled directly in this repo under
weights/weights_musclemap_wholebody.pth and loaded as this model's default
weights.

Run:
  python generate_musclemap_wholebody_model.py [output_dir]
"""

import os

INCREMENTAL_LEARN = False

if 'generate_convert' not in locals() and 'generate_convert' not in globals():
    from dafne_models.common import generate_convert

try:
    from dafne_dl import DynamicTorchModel
except ModuleNotFoundError:
    from dl import DynamicTorchModel


# ---------------------------------------------------------------------------
# Model architecture
# Parameters copied verbatim from the "wholebody" entry of MuscleMap's Zenodo
# model config (contrast_agnostic_wholebody_model.json, model version 1.4):
# spatial_dims=2, in_channels=1, out_channels=114 (113 muscles/bones +
# background), a 5-level UNet with LeakyReLU activations and instance norm.
# ---------------------------------------------------------------------------

def init_musclemap_wholebody_unet():
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=114,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        act='LeakyReLU',
        norm=Norm.INSTANCE,
    )
    return model


# ---------------------------------------------------------------------------
# apply_model_function
# All imports and constants must be local -- this function's source is
# serialized (via dill/inspect.getsource) and re-executed standalone, so
# nothing at module scope (including LABEL_MAP below) is available to it.
# ---------------------------------------------------------------------------

def musclemap_wholebody_apply(modelObj, data: dict):
    import numpy as np
    import torch
    from scipy.ndimage import zoom
    from scipy import ndimage as ndi
    from monai.inferers import sliding_window_inference

    # Output channel index (1..113) -> label name. Built from the sorted
    # list of unique label codes in contrast_agnostic_wholebody_model.json
    # v1.4 (region + anatomy + side), Title Case with _L/_R suffix, matching
    # MuscleMap's own remapping (id_map/inv_id_map + RemapLabels) applied
    # only for the wholebody model: the network is trained against
    # consecutive channel indices, not the raw sparse label codes (e.g.
    # 7101, 7102, ...).
    LABEL_MAP = {
        1: 'Levator Scapulae_L', 2: 'Levator Scapulae_R',
        3: 'Semispinalis Cervicis And Multifidus_L', 4: 'Semispinalis Cervicis And Multifidus_R',
        5: 'Semispinalis Capitis_L', 6: 'Semispinalis Capitis_R',
        7: 'Splenius Capitis_L', 8: 'Splenius Capitis_R',
        9: 'Sternocleidomastoid_L', 10: 'Sternocleidomastoid_R',
        11: 'Longus Colli_L', 12: 'Longus Colli_R',
        13: 'Trapezius_L', 14: 'Trapezius_R',
        15: 'Supraspinatus_L', 16: 'Supraspinatus_R',
        17: 'Subscapularis_L', 18: 'Subscapularis_R',
        19: 'Infraspinatus_L', 20: 'Infraspinatus_R',
        21: 'Deltoid_L', 22: 'Deltoid_R',
        23: 'Rhomboid_L', 24: 'Rhomboid_R',
        25: 'Thoracolumbar Multifidus_L', 26: 'Thoracolumbar Multifidus_R',
        27: 'Erector Spinae_L', 28: 'Erector Spinae_R',
        29: 'Psoas Major_L', 30: 'Psoas Major_R',
        31: 'Quadratus Lumborum_L', 32: 'Quadratus Lumborum_R',
        33: 'Latissimus Dorsi_L', 34: 'Latissimus Dorsi_R',
        35: 'Gluteus Minimus_L', 36: 'Gluteus Minimus_R',
        37: 'Gluteus Medius_L', 38: 'Gluteus Medius_R',
        39: 'Gluteus Maximus_L', 40: 'Gluteus Maximus_R',
        41: 'Tensor Fasciae Latae_L', 42: 'Tensor Fasciae Latae_R',
        43: 'Iliacus_L', 44: 'Iliacus_R',
        45: 'Ilium_L', 46: 'Ilium_R',
        47: 'Sacrum', 48: 'Femur_L',
        49: 'Femur_R', 50: 'Piriformis_L',
        51: 'Piriformis_R', 52: 'Pectineus_L',
        53: 'Pectineus_R', 54: 'Obturator Internus_L',
        55: 'Obturator Internus_R', 56: 'Obturator Externus_L',
        57: 'Obturator Externus_R', 58: 'Gemelli And Quadratus Femoris_L',
        59: 'Gemelli And Quadratus Femoris_R', 60: 'Vastus Lateralis_L',
        61: 'Vastus Lateralis_R', 62: 'Vastus Intermedius_L',
        63: 'Vastus Intermedius_R', 64: 'Vastus Medialis_L',
        65: 'Vastus Medialis_R', 66: 'Rectus Femoris_L',
        67: 'Rectus Femoris_R', 68: 'Sartorius_L',
        69: 'Sartorius_R', 70: 'Gracilis_L',
        71: 'Gracilis_R', 72: 'Semimembranosus_L',
        73: 'Semimembranosus_R', 74: 'Semitendinosus_L',
        75: 'Semitendinosus_R', 76: 'Biceps Femoris Long Head_L',
        77: 'Biceps Femoris Long Head_R', 78: 'Biceps Femoris Short Head_L',
        79: 'Biceps Femoris Short Head_R', 80: 'Adductor Magnus_L',
        81: 'Adductor Magnus_R', 82: 'Adductor Longus_L',
        83: 'Adductor Longus_R', 84: 'Adductor Brevis_L',
        85: 'Adductor Brevis_R', 86: 'Patella_L',
        87: 'Patella_R', 88: 'Tibialis Anterior_L',
        89: 'Tibialis Anterior_R', 90: 'Tibialis Posterior_L',
        91: 'Tibialis Posterior_R', 92: 'Peroneus Longus_L',
        93: 'Peroneus Longus_R', 94: 'Soleus_L',
        95: 'Soleus_R', 96: 'Medial Gastrocnemius_L',
        97: 'Medial Gastrocnemius_R', 98: 'Lateral Gastrocnemius_L',
        99: 'Lateral Gastrocnemius_R', 100: 'Tibia_L',
        101: 'Tibia_R', 102: 'Fibula_L',
        103: 'Fibula_R', 104: 'Flexor Hallucis Longus_L',
        105: 'Flexor Hallucis Longus_R', 106: 'Extensor Digitorum-Hallucis Longus_L',
        107: 'Extensor Digitorum-Hallucis Longus_R', 108: 'Flexor Digitorum Longus_L',
        109: 'Flexor Digitorum Longus_R', 110: 'Popliteus_L',
        111: 'Popliteus_R', 112: 'Plantaris_L',
        113: 'Plantaris_R',
    }
    # Pipeline mirrors MuscleMap's own pre_transforms/post_transforms in
    # mm_segment.py as closely as possible for a single 2D slice. Note
    # SpatialPadd's target is hardcoded to 256x256 in MuscleMap (matches the
    # wholebody roi_size here, but this constant is independent of ROI_SIZE
    # on principle -- see generate_musclemap_thigh_model.py's history for why),
    # and -- crucially -- Invertd (which undoes the resample, with linear
    # interpolation) runs BEFORE the argmax, i.e. MuscleMap resamples the
    # soft per-class network output back to native resolution and only then
    # discretizes, rather than discretizing at model resolution and
    # nearest-neighbor-upsampling the labels.
    MODEL_RESOLUTION = np.array([1.0, 1.0])
    ROI_SIZE = (256, 256)
    PAD_SIZE = (256, 256)
    FOREGROUND_MARGIN = 20  # matches CropForegroundd(margin=20) in MuscleMap
    OVERLAP_DEFAULT = 0.9  # matches mm_segment.py's --overlap default of 90
    MIN_MASK_PIXELS = 5  # drop spurious specks / empty masks from the output

    def fit_to_shape_chw(arr, shape):
        # Pad with zeros / crop (from the end) to force the last two
        # (spatial) dimensions of a (C, H, W) array to an exact shape,
        # compensating for rounding when a resample is inverted.
        pads = [(0, 0)] + [(0, max(0, s - arr.shape[i + 1])) for i, s in enumerate(shape)]
        if any(p[1] for p in pads):
            arr = np.pad(arr, pads, mode='constant')
        slices = (slice(None),) + tuple(slice(0, s) for s in shape)
        return arr[slices]

    # For a 2D model, Dafne's live app (MuscleSegmentation.getSegmentedMasks)
    # feeds data['image'] straight from the RAW, un-reoriented loaded volume
    # (self.imList, backed by dicomUtils' DicomReader) -- it does NOT call
    # ensure_compatible_orientation_inplace for 2D models (only the 3D path
    # does that). That raw per-slice layout does not match the RAS
    # convention MuscleMap trains/runs against: empirically (by comparing
    # against ground truth produced by the real mm_segment CLI, fed through
    # the actual Dafne loading path via dafne_models/bin/run_model.py with
    # the orientation-reformat step removed to mirror the live app) the
    # correct correction is a transpose followed by flipping both axes, not
    # a single-axis flip -- an earlier version of this plugin used
    # dafne_models/bin/run_model.py's own ensure_compatible_orientation_inplace
    # step as ground truth for "what Dafne feeds the model", which reformats
    # to metadata['orientation']='Axial' -- but the live app never does that
    # for 2D models, so that assumption was wrong and made real-app results
    # worse, not better. This transpose+double-flip is self-inverse in the
    # sense that applying the same recipe (transpose, then flip both axes)
    # to the corrected label map's OWN flipped-and-transposed form recovers
    # the original layout -- see the inverse at the end of this function.
    raw_image = np.asarray(data['image'], dtype=np.float32)
    raw_resolution = np.array(data['resolution'], dtype=np.float32)
    image = np.ascontiguousarray(raw_image.T[::-1, ::-1])
    resolution = raw_resolution[::-1]
    original_shape = image.shape

    # 1. Resample in-plane to the model's 1x1 mm training resolution
    # (order=1 linear, matching Spacingd's mode="bilinear")
    zoom_factor = resolution / MODEL_RESOLUTION
    resampled = zoom(image, zoom_factor, order=1)

    # 2. Normalize intensity using only nonzero voxels of the raw resampled
    # image (mirrors MONAI's NormalizeIntensityd(nonzero=True): background
    # stays exactly zero)
    raw_nonzero_mask = resampled != 0
    normalized = resampled.copy()
    if raw_nonzero_mask.any():
        mean = resampled[raw_nonzero_mask].mean()
        std = resampled[raw_nonzero_mask].std()
        normalized[raw_nonzero_mask] = (resampled[raw_nonzero_mask] - mean) / (std + 1e-8)

    # 3. Crop to the foreground bounding box with a margin. CropForegroundd
    # runs AFTER normalization in MuscleMap, with its default select_fn
    # (value > 0) evaluated on the already-normalized image -- not on the
    # pre-normalization nonzero mask.
    foreground_mask = normalized > 0
    rows = np.any(foreground_mask, axis=1)
    cols = np.any(foreground_mask, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        r0 = max(0, r0 - FOREGROUND_MARGIN)
        r1 = min(normalized.shape[0] - 1, r1 + FOREGROUND_MARGIN)
        c0 = max(0, c0 - FOREGROUND_MARGIN)
        c1 = min(normalized.shape[1] - 1, c1 + FOREGROUND_MARGIN)
    else:
        r0, r1, c0, c1 = 0, normalized.shape[0] - 1, 0, normalized.shape[1] - 1
    cropped = normalized[r0:r1 + 1, c0:c1 + 1]
    crop_shape = cropped.shape

    # 4. Pad up to the fixed 256x256 SpatialPadd target (method="end": pad
    # only at the bottom/right, never centered)
    pad_h = max(0, PAD_SIZE[0] - cropped.shape[0])
    pad_w = max(0, PAD_SIZE[1] - cropped.shape[1])
    padded = np.pad(cropped, ((0, pad_h), (0, pad_w)), mode='constant')

    # 5. Sliding-window inference with the bundled 2D UNet. Keep the raw
    # per-class logits (no argmax yet) -- MuscleMap resamples these back to
    # native resolution before discretizing (see step 7).
    model = modelObj.model
    model.eval()
    input_tensor = torch.as_tensor(padded[None, None, ...], dtype=torch.float32,
                                    device=modelObj.device)
    overlap = float(data.get('options', {}).get('overlap', OVERLAP_DEFAULT))
    with torch.no_grad():
        output = sliding_window_inference(input_tensor, ROI_SIZE, 1, model,
                                           overlap=overlap, mode='gaussian',
                                           device=modelObj.device)
    logits = output[0].cpu().numpy()  # (num_classes, Hp, Wp)

    # 6. Undo padding / crop, keeping all class channels
    logits = logits[:, :crop_shape[0], :crop_shape[1]]
    full_logits = np.zeros((logits.shape[0],) + normalized.shape, dtype=np.float32)
    full_logits[:, r0:r1 + 1, c0:c1 + 1] = logits

    # 7. Undo the resample with linear interpolation on the soft per-class
    # output (matches Invertd(nearest_interp=False) inverting Spacingd's
    # "bilinear" mode), THEN discretize -- mirrors MuscleMap running
    # Invertd before AsDiscreted(argmax=True) in its post_transforms.
    inv_zoom = 1.0 / zoom_factor
    resampled_back = zoom(full_logits, (1.0,) + tuple(inv_zoom), order=1)
    resampled_back = fit_to_shape_chw(resampled_back, original_shape)
    label_map = np.argmax(resampled_back, axis=0).astype(np.uint8)

    # 8. Keep only the largest connected component per label (2D per-slice
    # analog of MuscleMap's connected_chunks(), which keeps only the largest
    # connected component per label in the final reconstructed segmentation)
    structure = ndi.generate_binary_structure(2, 1)
    for label_value in np.unique(label_map):
        if label_value == 0:
            continue
        mask = label_map == label_value
        components, n_components = ndi.label(mask, structure=structure)
        if n_components > 1:
            counts = np.bincount(components.ravel())
            counts[0] = 0
            keep = counts.argmax()
            label_map[(components > 0) & (components != keep)] = 0

    # Undo the orientation correction from the top of this function (flip
    # both axes, then transpose -- the inverse of transpose-then-flip-both)
    # so masks line up with data['image']'s original layout.
    label_map = np.ascontiguousarray(label_map[::-1, ::-1].T)

    outputLabels = {}
    for label_value, label_name in LABEL_MAP.items():
        mask = (label_map == label_value).astype(np.uint8)
        if np.count_nonzero(mask) >= MIN_MASK_PIXELS:
            outputLabels[label_name] = mask
    return outputLabels


# ---------------------------------------------------------------------------
# incremental_learn_function
# ---------------------------------------------------------------------------

def musclemap_wholebody_incremental_learn(modelObj, trainingData: dict, trainingOutputs,
                                           bs=2, minTrainImages=5):
    import numpy as np
    import torch
    from scipy.ndimage import zoom
    from monai.losses import DiceCELoss

    LABEL_MAP = {
        1: 'Levator Scapulae_L', 2: 'Levator Scapulae_R',
        3: 'Semispinalis Cervicis And Multifidus_L', 4: 'Semispinalis Cervicis And Multifidus_R',
        5: 'Semispinalis Capitis_L', 6: 'Semispinalis Capitis_R',
        7: 'Splenius Capitis_L', 8: 'Splenius Capitis_R',
        9: 'Sternocleidomastoid_L', 10: 'Sternocleidomastoid_R',
        11: 'Longus Colli_L', 12: 'Longus Colli_R',
        13: 'Trapezius_L', 14: 'Trapezius_R',
        15: 'Supraspinatus_L', 16: 'Supraspinatus_R',
        17: 'Subscapularis_L', 18: 'Subscapularis_R',
        19: 'Infraspinatus_L', 20: 'Infraspinatus_R',
        21: 'Deltoid_L', 22: 'Deltoid_R',
        23: 'Rhomboid_L', 24: 'Rhomboid_R',
        25: 'Thoracolumbar Multifidus_L', 26: 'Thoracolumbar Multifidus_R',
        27: 'Erector Spinae_L', 28: 'Erector Spinae_R',
        29: 'Psoas Major_L', 30: 'Psoas Major_R',
        31: 'Quadratus Lumborum_L', 32: 'Quadratus Lumborum_R',
        33: 'Latissimus Dorsi_L', 34: 'Latissimus Dorsi_R',
        35: 'Gluteus Minimus_L', 36: 'Gluteus Minimus_R',
        37: 'Gluteus Medius_L', 38: 'Gluteus Medius_R',
        39: 'Gluteus Maximus_L', 40: 'Gluteus Maximus_R',
        41: 'Tensor Fasciae Latae_L', 42: 'Tensor Fasciae Latae_R',
        43: 'Iliacus_L', 44: 'Iliacus_R',
        45: 'Ilium_L', 46: 'Ilium_R',
        47: 'Sacrum', 48: 'Femur_L',
        49: 'Femur_R', 50: 'Piriformis_L',
        51: 'Piriformis_R', 52: 'Pectineus_L',
        53: 'Pectineus_R', 54: 'Obturator Internus_L',
        55: 'Obturator Internus_R', 56: 'Obturator Externus_L',
        57: 'Obturator Externus_R', 58: 'Gemelli And Quadratus Femoris_L',
        59: 'Gemelli And Quadratus Femoris_R', 60: 'Vastus Lateralis_L',
        61: 'Vastus Lateralis_R', 62: 'Vastus Intermedius_L',
        63: 'Vastus Intermedius_R', 64: 'Vastus Medialis_L',
        65: 'Vastus Medialis_R', 66: 'Rectus Femoris_L',
        67: 'Rectus Femoris_R', 68: 'Sartorius_L',
        69: 'Sartorius_R', 70: 'Gracilis_L',
        71: 'Gracilis_R', 72: 'Semimembranosus_L',
        73: 'Semimembranosus_R', 74: 'Semitendinosus_L',
        75: 'Semitendinosus_R', 76: 'Biceps Femoris Long Head_L',
        77: 'Biceps Femoris Long Head_R', 78: 'Biceps Femoris Short Head_L',
        79: 'Biceps Femoris Short Head_R', 80: 'Adductor Magnus_L',
        81: 'Adductor Magnus_R', 82: 'Adductor Longus_L',
        83: 'Adductor Longus_R', 84: 'Adductor Brevis_L',
        85: 'Adductor Brevis_R', 86: 'Patella_L',
        87: 'Patella_R', 88: 'Tibialis Anterior_L',
        89: 'Tibialis Anterior_R', 90: 'Tibialis Posterior_L',
        91: 'Tibialis Posterior_R', 92: 'Peroneus Longus_L',
        93: 'Peroneus Longus_R', 94: 'Soleus_L',
        95: 'Soleus_R', 96: 'Medial Gastrocnemius_L',
        97: 'Medial Gastrocnemius_R', 98: 'Lateral Gastrocnemius_L',
        99: 'Lateral Gastrocnemius_R', 100: 'Tibia_L',
        101: 'Tibia_R', 102: 'Fibula_L',
        103: 'Fibula_R', 104: 'Flexor Hallucis Longus_L',
        105: 'Flexor Hallucis Longus_R', 106: 'Extensor Digitorum-Hallucis Longus_L',
        107: 'Extensor Digitorum-Hallucis Longus_R', 108: 'Flexor Digitorum Longus_L',
        109: 'Flexor Digitorum Longus_R', 110: 'Popliteus_L',
        111: 'Popliteus_R', 112: 'Plantaris_L',
        113: 'Plantaris_R',
    }
    NAME_TO_VALUE = {name: value for value, name in LABEL_MAP.items()}
    MODEL_RESOLUTION = np.array([1.0, 1.0])
    TRAIN_SIZE = (256, 256)  # matches ROI_SIZE, divisible by 16 (4 strides of 2)
    EPOCHS = 5
    LEARNING_RATE = 1e-4

    def fit_to_shape(arr, shape, mode='constant'):
        out = arr
        pads = [(0, max(0, s - out.shape[i])) for i, s in enumerate(shape)]
        if any(p[1] for p in pads):
            out = np.pad(out, pads, mode=mode)
        slices = tuple(slice(0, s) for s in shape)
        return out[slices]

    image_list = trainingData['image_list']
    # Dafne feeds 2D models raw, un-reoriented slices (see the matching,
    # more detailed comment in musclemap_wholebody_apply) -- transpose +
    # flip both axes to align with the RAS convention the bundled backbone
    # was pretrained on, so incrementally-learned weights stay consistent
    # with the pretrained ones. resolution is swapped to match the transpose.
    raw_resolution = np.array(trainingData['resolution'], dtype=np.float32)
    resolution = raw_resolution[::-1]
    n_images = len(image_list)

    if n_images < minTrainImages:
        print(f'MuscleMap wholebody: not enough images for incremental learning '
              f'({n_images} < {minTrainImages})')
        return

    zoom_factor = resolution / MODEL_RESOLUTION

    images_out = []
    labels_out = []
    for image, label_dict in zip(image_list, trainingOutputs):
        image = np.ascontiguousarray(np.asarray(image, dtype=np.float32).T[::-1, ::-1])

        # build a single integer label map (consecutive channel indices, not
        # MuscleMap's raw sparse codes) from the per-name binary masks
        label_img = np.zeros(image.shape, dtype=np.int64)
        for name, mask in label_dict.items():
            value = NAME_TO_VALUE.get(name)
            if value is not None:
                mask_t = np.ascontiguousarray(np.asarray(mask).T[::-1, ::-1])
                label_img[mask_t > 0] = value

        resampled = zoom(image, zoom_factor, order=1)
        resampled_label = zoom(label_img, zoom_factor, order=0)

        nonzero_mask = resampled != 0
        normalized = resampled.copy()
        if nonzero_mask.any():
            mean = resampled[nonzero_mask].mean()
            std = resampled[nonzero_mask].std()
            normalized[nonzero_mask] = (resampled[nonzero_mask] - mean) / (std + 1e-8)

        normalized = fit_to_shape(normalized, TRAIN_SIZE)
        resampled_label = fit_to_shape(resampled_label, TRAIN_SIZE)

        images_out.append(normalized)
        labels_out.append(resampled_label)

    model = modelObj.model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

    images_t = torch.as_tensor(np.stack(images_out)[:, None, ...], dtype=torch.float32)
    labels_t = torch.as_tensor(np.stack(labels_out)[:, None, ...], dtype=torch.long)

    batch_size = max(1, bs)
    n = images_t.shape[0]

    print(f'MuscleMap wholebody: incremental learning on {n} images for {EPOCHS} epochs')
    for epoch in range(EPOCHS):
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            batch_images = images_t[idx].to(modelObj.device)
            batch_labels = labels_t[idx].to(modelObj.device)

            optimizer.zero_grad()
            output = model(batch_images)
            loss = loss_function(output, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        print(f'  epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss / max(1, n_batches):.4f}')

    model.eval()


metadata = {
    'categories': [['MSK', 'Muscle', 'Whole body']],
    'variants': [''],
    'dimensionality': '2',
    'model_name': 'MuscleMap Wholebody',
    'model_type': 'DynamicTorchModel',
    'orientation': 'Axial',
    'info': {
        'Description': 'Contrast-agnostic segmentation of 113 muscles/bones across the whole '
                        'body (neck to foot), reimplementing the MuscleMap toolbox 2D UNet '
                        '(v1.4, the actively maintained MuscleMap model) as a native Dafne model.',
        'Author': 'MuscleMap team (Yin, Wesselink, Kim, Weber et al.); Dafne reimplementation',
        'Modality': 'MRI, CT',
        'Reference': 'https://github.com/MuscleMap/MuscleMap',
        'Link': 'https://zenodo.org/records/21929873',
    },
    'dependencies': {
        'monai': 'monai >= 1.3.0',
        'scipy': 'scipy',
    },
    'options': {
        'overlap': 'float',  # sliding-window inference overlap fraction, default 0.9
    },
}

generate_convert(
    model_id='609e8235-ae45-496b-aa43-e6c892d03c63',
    default_weights_path=os.path.join('weights', 'weights_musclemap_wholebody.pth'),
    model_name_prefix='MuscleMap_Wholebody',
    model_create_function=init_musclemap_wholebody_unet,
    model_apply_function=musclemap_wholebody_apply,
    model_learn_function=musclemap_wholebody_incremental_learn if INCREMENTAL_LEARN else None,
    dimensionality=2,
    model_type=DynamicTorchModel,
    metadata=metadata,
)
