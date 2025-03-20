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
import os
import torch

join=os.path.join

if 'generate_convert' not in locals() and 'generate_convert' not in globals():
    from dafne_models.common import generate_convert

from dafne_dl.DynamicEnsembleModel import DynamicEnsembleModel

def init_folds():
    from monai.networks.nets import SwinUNETR, UNETR, DynUNet

    model_fold0 = SwinUNETR(
        feature_size= 48,
        img_size=(128,128,128),
        in_channels=1,
        out_channels=2,
        spatial_dims= 3,
        use_checkpoint= False,
        use_v2= False,
    )

    model_fold1 = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size= [3, [1, 1, 3], 3, 3],
        strides= [1, 2, 2, 1],
        upsample_kernel_size= [2, 2, 1],
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision= False,
        deep_supr_num= 1,
        res_block= False,
    )

    model_fold2 = UNETR(
        feature_size= 16,
        img_size=(128,128,128),
        in_channels=1,
        out_channels=2,
        # spatial_dims: 3
        hidden_size= 768,
        mlp_dim= 3072,
        num_heads= 12,
        proj_type= "conv",
        norm_name= "instance",
        res_block= True,
        dropout_rate= 0.0,
        # use_checkpoint: True
    )

    model_fold3 = SwinUNETR(
        feature_size= 48,
        img_size=(128,128,128),
        in_channels=1,
        out_channels=2,
        spatial_dims= 3,
        use_checkpoint= False,
        use_v2= False,
    )
    
    model_fold4 = SwinUNETR(
        feature_size= 48,
        img_size=(128,128,128),
        in_channels=1,
        out_channels=2,
        spatial_dims= 3,
        use_checkpoint= False,
        use_v2= False,
    )

    return model_fold0, model_fold1, model_fold2, model_fold3, model_fold4

def ensemble_apply(modelObj, data: dict):
    try:
        import dafne_dl.common.preprocess_train as pretrain 
    except ModuleNotFoundError:
        import dl.common.preprocess_train as pretrain 
    from dafne_dl.interfaces import WrongDimensionalityError
    import os
    import gc
    import numpy as np
    from monai.transforms import (
        Compose,
        # LoadImaged,
        EnsureChannelFirstd,
        EnsureTyped,
        Orientationd,
        Spacingd,
        # SpatialResample,
        CastToTyped,
        ScaleIntensityd,
        Invertd,
        Activationsd,
        CopyItemsd,
        AsDiscreted,
    )
    from monai.inferers import sliding_window_inference
    from monai.utils import set_determinism
    from monai.data import MetaTensor, Dataset, DataLoader, decollate_batch
    import torch

    set_determinism(seed=0)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,max_split_size_mb:32,expandable_segments:True"

    # Parameters
    MODEL_RESOLUTION = np.array([1.0, 1.0, 1.0])
    roi_size = (128,128,128)
    sw_batch_size = 1

    # data = pretrain.common_input_process_ensemble_inference(data, MODEL_RESOLUTION)

    if len(data['image'].shape) != 3:
        raise WrongDimensionalityError("Input image must be 3D")

    if modelObj.device.type == 'cpu':
        device = 'cpu'
    else:
        if modelObj.device.index is not None:
            device = modelObj.device.index
        else:
            device = 0

    # Define data transforms
    pre_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear"), align_corners=True),
            CastToTyped(keys=["image"], dtype= np.float32),
            ScaleIntensityd(keys=["image"]),
            CastToTyped(keys=["image"], dtype= np.float32),
        ]
    )

    post_pred=Compose(
        [
            Invertd(
                keys="pred",
                transform=pre_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            Activationsd(keys="pred", softmax=True, sigmoid=False),
            CopyItemsd(keys="pred", times=1, names="pred_final"),
            AsDiscreted(keys="pred_final", argmax=True), #argmax=True
        ]
    )

    # data loading

    image = data['image']
    affine=data['affine']

    if image.ndim == 3: 
        image = np.expand_dims(image, axis=0)

    torch_data = [
        {
            "image" : MetaTensor(image, affine = affine),
        },
    ]
    input_data = Dataset(data=torch_data, transform=pre_transforms)
    input_tensor = DataLoader(input_data, batch_size=1, num_workers=0)

    torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():

        seg_all=[]

        for ii in range(5):

            torch.cuda.empty_cache()
            gc.collect()

            model_load = modelObj.model[ii].to(device)
            model_load.eval()
            print('model loaded')

            test_data = list(input_tensor)[0]
            test_input = test_data["image"].to(device)

            with torch.no_grad():

                test_data["pred"] = sliding_window_inference(test_input, roi_size, sw_batch_size, model_load, mode="gaussian", overlap=0.5)
                print("Model applied")

            # Post-processing
            val_outputs_list = decollate_batch(test_data)

            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            val_output=val_output_convert[0]["pred_final"]

            model_load.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()
            del model_load

            torch.cuda.empty_cache()
            gc.collect()

            print('val_output shape: ', val_output.shape)

            seg = np.squeeze(val_output[0, ...]) 
            seg_all.append(seg)

            del val_output, val_output_convert, val_outputs_list, test_data, test_input
            torch.cuda.empty_cache()
            gc.collect()

        # Ensemble predictions
        seg_all = [torch.tensor(arr) for arr in seg_all]
        summed_voxel=seg_all[0]
        for i in range(1, 5):
            summed_voxel=torch.add(summed_voxel, seg_all[i])

        ensemble=torch.where(summed_voxel>2, 1, 0).cpu().numpy()

    return {
        'CHP': ensemble.astype(np.uint8),
    }


def ensemble_incremental_learning(modelObj, trainingData: dict, trainingOutputs, bs=1, minTrainImages=10):
    try:
        import dafne_dl.common.preprocess_train as pretrain 
        from dafne_dl.labels.chp import inverse_labels 
    except ModuleNotFoundError:
        import dl.common.preprocess_train as pretrain 
        from dl.labels.chp import inverse_labels 

    import os
    import gc
    import time
    import math
    # import pickle as pkl
    from tqdm import tqdm
    import torch
    import torch.distributed as dist
    import torchio.transforms as torchio_transforms
    from torch.amp import GradScaler, autocast
    # import torch.utils.checkpoint as checkpoint
    from dafne.utils import compressed_pickle
    # import monai
    import monai.transforms as monai_transforms
    from monai import losses
    from monai.data import DataLoader, MetaTensor, CacheDataset
    from monai.metrics import DiceMetric
    from monai.utils import set_determinism
    from monai.inferers import sliding_window_inference
    from monai.data import decollate_batch

    try:
        np
    except:
        import numpy as np

    # torch.cuda.empty_cache()
    # gc.collect()
    # torch.cuda.ipc_collect()  
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,max_split_size_mb:32,expandable_segments:True"

    set_determinism(seed=0)
    

    # # validation
    # def validation(epoch_iterator_val, global_step, model_, device, patch):

    #     patch = patch
    #     softmax = True
    #     output_classes = 2
    #     overlap_ratio = 0.5 #0.8
    #     sw_batch_size = 1

    #     dice_metric = DiceMetric(include_background=False, reduction="mean")


    #     # post transforms
    #     if softmax:
    #         post_pred = monai_transforms.Compose(
    #             [monai_transforms.EnsureType(), monai_transforms.AsDiscrete(argmax=True, to_onehot=output_classes)]
    #         )
    #         post_label = monai_transforms.Compose([monai_transforms.EnsureType(), monai_transforms.AsDiscrete(to_onehot=output_classes)])
    #     else:
    #         post_pred = monai_transforms.Compose(
    #             [monai_transforms.EnsureType(), monai_transforms.Activations(sigmoid=True), monai_transforms.AsDiscrete(threshold=0.5)]
    #         )

    #     model_.eval()
    #     dice_vals = list()

    #     with torch.no_grad():
    #         for step, batch in enumerate(epoch_iterator_val):

    #             val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))

    #             # torch.cuda.empty_cache()
    #             # gc.collect()

    #             val_outputs = sliding_window_inference(val_inputs, patch, sw_batch_size, model_, overlap=overlap_ratio)

    #             torch.cuda.empty_cache()
    #             gc.collect()

    #             val_labels_list = decollate_batch(val_labels)
    #             val_labels_convert = [
    #                 post_label(val_label_tensor) for val_label_tensor in val_labels_list
    #             ]
    #             # val_labels_convert = post_label(val_labels)

    #             val_outputs_list = decollate_batch(val_outputs)
    #             val_output_convert = [
    #                 post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
    #             ]

    #             # val_output_convert = post_pred(val_outputs)

    #             #Dice
    #             dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    #             dice = dice_metric.aggregate().item()
    #             dice_vals.append(dice)

    #             epoch_iterator_val.set_description(
                
    #                 "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice) 
    #             )
    #             del val_inputs, val_labels, val_outputs,  val_labels_convert, val_output_convert, val_labels_list, val_outputs_list
                
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             # torch.cuda.ipc_collect()
    #         dice_metric.reset()

    #     mean_dice_val = np.mean(dice_vals)

    #     metrics_values=mean_dice_val
        
    #     return metrics_values  

    # validation
    def validation(val_loader, global_step, model_, device, patch):

        patch = patch
        softmax = True
        output_classes = 2
        overlap_ratio = 0.5

        dice_metric = DiceMetric(include_background=False, reduction="mean")


        # post transforms
        if softmax:
            post_pred = monai_transforms.Compose(
                [monai_transforms.EnsureType(), monai_transforms.AsDiscrete(argmax=True, to_onehot=output_classes)]
            )
            post_label = monai_transforms.Compose([monai_transforms.EnsureType(), monai_transforms.AsDiscrete(to_onehot=output_classes)])
        else:
            post_pred = monai_transforms.Compose(
                [monai_transforms.EnsureType(), monai_transforms.Activations(sigmoid=True), monai_transforms.AsDiscrete(threshold=0.5)]
            )

        model_.eval()
        dice_vals = list()

        with torch.autograd.set_grad_enabled(False):
        # with torch.no_grad(), torch.set_grad_enabled(False), torch.amp.autocast("cuda"):
            for batch in val_loader:

                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))

                torch.cuda.empty_cache()
                gc.collect()

                with torch.no_grad():

                    val_outputs["pred"] = sliding_window_inference(val_inputs, patch, 1, model_, overlap=overlap_ratio)

                torch.cuda.empty_cache()
                gc.collect()
                # torch.cuda.ipc_collect()

                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                # val_labels_convert = post_label(val_labels)

                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]

                # val_output_convert = post_pred(val_outputs)

                #Dice
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)

                del val_inputs, val_labels, val_outputs,  val_labels_convert, val_output_convert, val_labels_list, val_outputs_list
                
                torch.cuda.empty_cache()
                gc.collect()
            dice_metric.reset()

        mean_dice_val = np.mean(dice_vals)

        metrics_values=mean_dice_val
        
        return metrics_values  

    MODEL_RESOLUTION = np.array([1.0, 1.0, 1.0])
    MODEL_SIZE = (128,128,128)
    BATCH_SIZE = bs
    # BAND = 64
    MIN_TRAINING_IMAGES = minTrainImages

    if modelObj.device.type == 'cpu':
        device = 'cpu'
    else:
        if modelObj.device.index is not None:
            device = modelObj.device.index
        else:
            device = 0


    t = time.time()

    train_transforms = monai_transforms.Compose(
        [
            monai_transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim=0), #channel_dim='no_channel'
            monai_transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai_transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), align_corners=(True, True)),
            monai_transforms.SpatialPadd(keys=["image", "label"], spatial_size=MODEL_SIZE),
            monai_transforms.CastToTyped(keys=["image"], dtype= np.float32),
            monai_transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            monai_transforms.EnsureTyped(keys=["image", "label"]),
            monai_transforms.RandBiasFieldd(keys=["image"], degree=3, prob=0.15),
            monai_transforms.RandGibbsNoised(keys=["image"], prob=0.15),
            torchio_transforms.RandomMotion(
                include=["image"], 
                label_keys=["label"], 
                degrees=10, 
                translation=10, 
                num_transforms=2, 
                image_interpolation=("linear"), 
                p=0.3
            ),
            torchio_transforms.RandomElasticDeformation(
                include=["image"], 
                label_keys=["label"], 
                num_control_points= 7,
                max_displacement= 7.5,
                locked_borders= 2,
                image_interpolation=("linear"),
                label_interpolation=("nearest"),
                p=0.3
            ),
            torchio_transforms.RandomGhosting(
                include=["image"], 
                label_keys=["label"], 
                num_ghosts= 10,
                axes= [0, 1, 2],
                intensity= [0.5, 1],
                restore= 0.2,
                p= 0.3
            ),
            monai_transforms.RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=0),
            monai_transforms.RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=1),
            monai_transforms.RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=2),
            monai_transforms.RandRotated(keys=["image", "label"], mode=("bilinear", "nearest"), prob=0.15, range_x=0.3, range_y=0.3, range_z=0.3),
            monai_transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
            monai_transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=MODEL_SIZE,
                pos= 1.0,
                neg= 0.0    
            ),
            monai_transforms.SpatialPadd(keys=["image", "label"], mode=('reflect', 'constant'), spatial_size=MODEL_SIZE),
            monai_transforms.CastToTyped(keys=["image", "label"], dtype= (np.float32, np.uint8)),
        ]
    )

    val_transforms = monai_transforms.Compose(
        [
            monai_transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim=0), #  channel_dim='no_channel'
            monai_transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai_transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), align_corners=(True, True)),
            monai_transforms.CastToTyped(keys=["image"], dtype= np.float32),
            monai_transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            monai_transforms.CastToTyped(keys=["image", "label"], dtype= (np.float32, np.uint8)),
        ]
    )

    post_pred = monai_transforms.Compose(
        [monai_transforms.EnsureType(), monai_transforms.AsDiscrete(argmax=True, to_onehot=2)]
    )
    post_label = monai_transforms.Compose([monai_transforms.EnsureType(), monai_transforms.AsDiscrete(to_onehot=2)])

    image_list, mask_list = pretrain.common_input_process_ensemble(inverse_labels, MODEL_RESOLUTION, trainingData, trainingOutputs)

    affine_list = trainingData['affine']

    print('Done. Elapsed', time.time() - t)
    nImages = len(image_list)

    if nImages < MIN_TRAINING_IMAGES:
        print("Not enough images for training")
        return

    # print('Weight calculation')
    # t = time.time()

    train_files=[]
    validation_files=[]

    jj=math.ceil(3*len(image_list)/4)

    for kk in range(len(image_list)):

            image=image_list[kk]
            print('image shape: ', image.shape)
            seg=mask_list[kk]
            print('seg shape: ', seg.shape)
            affine_=affine_list[kk]

            if image.ndim == 3: 
                image = np.expand_dims(image, axis=0)
                seg = np.expand_dims(seg, axis=0)

            if kk>jj:
                validation_files.append({"image": MetaTensor(image, affine = affine_), "label": MetaTensor(seg, affine = affine_)})
            else:
                train_files.append({"image": MetaTensor(image, affine = affine_), "label": MetaTensor(seg, affine = affine_)})
    
    # print('Done. Elapsed', time.time() - t)

    print(f'Incremental learning for Choroid Plexus with {nImages} images')
    t = time.time()
    
    
    train_ds = CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=float(torch.cuda.device_count()) / 4.0,
        num_workers=8,
        progress=False,
        )
    val_ds = CacheDataset(
        data=validation_files, 
        transform=val_transforms, 
        cache_rate=float(torch.cuda.device_count()) / 4.0,
        num_workers=2,
        progress = False,
        )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle = False)

    torch.cuda.empty_cache()
    gc.collect()
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # loss_function=losses.DiceCELoss(include_background=False,to_onehot_y=2, sigmoid=False)

    # max_iterations= 10 #10000
    # num_iterations_per_validation= 2 #100
    # num_epochs_per_validation = num_iterations_per_validation // len(train_files)
    # num_epochs_per_validation = max(num_epochs_per_validation, 1)
    # # num_epochs = num_epochs_per_validation * (max_iterations // num_iterations_per_validation)

    # state_dicts=[]

    # for ii in range(5):

    #     set_determinism(seed=0)
        
    #     print(f"Model {ii} load")

    #     model_ = modelObj.model[ii].to(device)

    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()

    #     optimizer = torch.optim.AdamW(model_.parameters(), lr=0.0001, weight_decay= 1.0e-05) 

    #     # training
    #     global_step = 0
    #     dice_val_best = 0.0
    #     global_step_best = 0

    #     while global_step < max_iterations: 

    #         model_.train()
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()
    #         epoch_loss = 0
    #         step = 0
    #         epoch_iterator = tqdm(
    #             train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    #         )

    #         for step, batch in enumerate(epoch_iterator):

    #             step += 1
    #             print(f'step {step}')
    #             x, y = (batch["image"].to(device), batch["label"].to(device))
    #             # print('x shape: ', x.shape)
    #             # print('y shape: ', y.shape)
    #             print('x and y')

    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             torch.cuda.ipc_collect()

    #             logit_map = model_(x)
    #             print('logit map')
    #             loss = loss_function(logit_map, y) 
    #             # print(torch.cuda.memory_summary())
    #             # logit_map = model_(x)
    #             # loss = loss_function(logit_map, y) 
    #             print('loss')

    #             loss.backward()
    #             print('loss backward')


    #             epoch_loss += loss.item()
    #             print('epoch loss')

    #             del logit_map, x, y

    #             optimizer.step()

    #             print('optimizer')
    #             optimizer.zero_grad()
    #             epoch_iterator.set_description(
    #                 "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
    #             )

    #             del loss

    #             torch.cuda.empty_cache()
    #             torch.cuda.ipc_collect()

    #             if (
    #                 global_step % num_iterations_per_validation == 0 and global_step != 0
    #             ) or global_step == max_iterations:
    #                 epoch_iterator_val = tqdm(
    #                     val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
    #                 )

    #                 dice_val=validation(epoch_iterator_val, global_step, model_, device, MODEL_SIZE)
    #                 epoch_loss /= step


    #                 torch.cuda.empty_cache()
    #                 gc.collect()
    #                 # torch.cuda.ipc_collect()
                    
    #                 if (dice_val > dice_val_best) : 
                        
    #                     dice_val_best = dice_val
    #                     global_step_best = global_step

    #                     last_good_weights = compressed_pickle.dumps(model_.state_dict()) 

    #                     print(
    #                         f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best:.4f} \n"
    #                     )
    #                 else:
    #                     print(
    #                         f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best:.4f},  Current Avg. Dice: {dice_val:.4f} \n"
    #                     )

    #             global_step += 1
                
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #             # torch.cuda.ipc_collect()
        
    #     del optimizer, model_
        
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     # torch.cuda.ipc_collect()

    #     new_state_dict = compressed_pickle.loads(last_good_weights)
    #     state_dicts.append(new_state_dict)

    loss_function=losses.DiceCELoss(include_background=False,to_onehot_y=2, sigmoid=False)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_iterations= 10 #10000
    num_iterations_per_validation= 2 #100
    num_epochs_per_validation = num_iterations_per_validation // len(train_files)
    num_epochs_per_validation = max(num_epochs_per_validation, 1)
    num_epochs = num_epochs_per_validation * (max_iterations // num_iterations_per_validation)


    state_dicts=[]

    for ii in range(5):

        set_determinism(seed=0)
        
        print(f"Model {ii} load")

        model_ = modelObj.model[ii].to(device)

        # scaler
        scaler = GradScaler("cuda")

        print("num_epochs", num_epochs)
        print("num_epochs_per_validation", num_epochs_per_validation)

        # optimizer
        optimizer = torch.optim.AdamW(model_.parameters(), lr=0.0001, weight_decay= 1.0e-05) 

        # training
        val_interval = num_epochs_per_validation
        global_step = 0
        dice_val_best = 0.0
        global_step_best = -1

        gc.collect()
        torch.cuda.empty_cache()

        for epoch in range(num_epochs):

        # while global_step < max_iterations: 
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}")

            model_.train()
            gc.collect()
            torch.cuda.empty_cache()

            epoch_loss = 0
            loss_torch = torch.zeros(2, dtype=torch.float, device=device)

            gc.collect()
            torch.cuda.empty_cache()

            step = 0

            # epoch_iterator = tqdm(
            #     train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
            # )

            # accumulation_steps = 4  # Numero di passaggi da accumulare prima di fare una retropropagazione

            optimizer.zero_grad()  # Azzera i gradienti iniziali
            # scaler = GradScaler("cuda")
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.ipc_collect()
            
            for batch_data in train_loader:
            # for step, batch in enumerate(epoch_iterator):

                step += 1
                print(f'step {step}')
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.ipc_collect()

                for param in model_.parameters():
                    param.grad = None

                print('model_params')
                
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.ipc_collect()

                
                with torch.amp.autocast('cuda'):
                    print('enter')
                    outputs = model_(inputs)
                    print('logit map')
                    loss = loss_function(outputs.float(), labels) 
                    print('loss')
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                loss_torch[0] += loss.item()
                loss_torch[1] += 1.0
                epoch_len = len(train_loader)

                global_step += 1

                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

                # loss_torch = loss_torch.tolist()
                # loss_torch_epoch = loss_torch[0] / loss_torch[1]
                
                loss_torch_list = loss_torch.tolist()

                loss_torch_epoch = loss_torch_list[0] / loss_torch_list[1]

                print(
                    f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, "
                    f"best mean dice: {dice_val_best:.4f} at epoch {global_step_best}"
                )

                if (epoch + 1) % val_interval == 0 or (epoch + 1) == num_epochs:

                    torch.cuda.empty_cache()
                    model_.eval()

                    with torch.no_grad():
                        _index = 0
                        dice_vals = list()

                        for val_data in val_loader:

                            val_images = val_data["image"].to(device)
                            val_labels = val_data["label"].to(device)
                            
                            with torch.amp.autocast('cuda'):
                                val_outputs = sliding_window_inference(
                                    val_images,
                                    MODEL_SIZE,
                                    1,
                                    model_,
                                    overlap=0.5,
                                )
                            val_outputs = post_pred(val_outputs[0, ...])
                            val_outputs = val_outputs[None, ...]
                            val_labels = post_label(val_labels[0, ...])
                            val_labels = val_labels[None, ...]

                            dice_metric(y_pred=val_outputs, y=val_labels)
                            dice = dice_metric.aggregate().item()
                            dice_vals.append(dice)

                            _index += 1
                        dice_metric.reset()
                    dice_val = np.mean(dice_vals)

                    # dice_val=validation(val_loader, global_step, model_, device, MODEL_SIZE)
                    
                    if (dice_val > dice_val_best) : 
                        
                        dice_val_best = dice_val
                        global_step_best = epoch +1 

                        last_good_weights = compressed_pickle.dumps(model_.state_dict()) 

                        print(
                            f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best:.4f} \n"
                        )
                    else:
                        print(
                            f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best:.4f},  Current Avg. Dice: {dice_val:.4f} \n"
                        )
                    print(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, dice_val, dice_val_best, global_step_best
                        )
                    )
                
                torch.cuda.empty_cache()
               
        del optimizer, model_
        print(f"train completed, best Dice metric: {dice_val_best:.4f} at epoch: {global_step_best}")
        
        torch.cuda.empty_cache()
        gc.collect()


        new_state_dict = compressed_pickle.loads(last_good_weights)
        state_dicts.append(new_state_dict)

        
        torch.cuda.empty_cache()


    print('Done. Elapsed', time.time() - t)  

    for ii in range(0, len(state_dicts)):
        modelObj.model[ii].load_state_dict(state_dicts[ii])


metadata = {
    'dependencies': {
        'monai': 'monai = 1.4.0',
        'torchio': 'torchio = 0.20.1',
    },
    'description': 'This model segments Choroid Plexus from T1-w MRI images based on ASCHOPLEX.',
}

info_json = {
    'categories': ['CHP'],
    'variants': [""],
    'dimensionality': "3",
    'model_name': 'aschoplex',
    'model_type': 'DynamicEnsembleModel',
    'info': {
        'Description': 'ASCHOPLEX: automatic segmentation of Choroid Plexus model',
        'Author':	'Visani Valentina',
        'Modality': 'MRI',
        'Orientation': '', #'Axial',
        "Link": '',
    },
    }
    

generate_convert(model_id='e2bb676f-6e8e-45b8-b5d7-542ef8f3e542',
                 default_weights_path='weights',
                 model_name_prefix='aschoplex',
                 model_create_function=init_folds,
                 model_apply_function=ensemble_apply,
                 model_learn_function=ensemble_incremental_learning,
                 dimensionality=3,
                 model_type=DynamicEnsembleModel,
                 metadata=metadata,
                 info_json = info_json
                 )
