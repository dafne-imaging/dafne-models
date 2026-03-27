import os
import shutil
import sys
import dill

from dafne_dl import DynamicDLModel

def _is_tf_model(model):
    if hasattr(model, 'load_weights'):
        return True
    return False

def _is_tf_ensemble_model(model):
    if isinstance(model, list) and all(hasattr(m, 'load_weights') for m in model):
        return True
    return False

def _is_torch_model(model):
    if hasattr(model, 'load_state_dict'):
        return True
    return False

def _is_torch_ensemble_model(model):
    if isinstance(model, list) and all(hasattr(m, 'load_state_dict') for m in model):
        return True
    return False


def generate_convert(model_id,
                     default_weights_path,
                     model_name_prefix,
                     model_create_function,
                     model_apply_function,
                     model_learn_function,
                     data_preprocess_function = None,  # function that preprocesses the data
                     data_postprocess_function = None,  # function that postprocesses the data
                     label_preprocess_function = None,  # function that preprocesses the labels for training
                     dimensionality=2,
                     model_type=DynamicDLModel,
                     metadata=None):
    """
    Function that either generates a new model using the default weights, or updates an existing model, based on argv[1]

    Parameters:
        model_id: the id of the model to be generated/updated
        default_weights_path: the path to the default weights
        model_name_prefix: the prefix of the model name (e.g. 'Leg' or 'Thigh'). Used for the filename
        model_create_function: the function that creates the model
        model_apply_function: the function that applies the model
        model_learn_function: the function that performs the incremental learning of the model
        dimensionality: the dimensionality of the data (2 or 3)
        model_type: the type of the model (e.g. DynamicDLModel or DynamicTorchModel or DynamicEnsembleModel)
        metadata: additional metadata to be stored in the model

    Returns:
        None
    """
    if not metadata:
        metadata = {}
    weights = None
    existing_model = False
    base_folder = 'models'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        base_folder = os.path.dirname(filename)
        if filename.endswith('.model'):
            # convert an existing model
            print("Converting model", sys.argv[1])
            old_model_path = sys.argv[1]
            filename = old_model_path
            old_model = model_type.Load(open(old_model_path, 'rb'))
            shutil.move(old_model_path, old_model_path + '.bak')
            weights = old_model.get_weights()
            timestamp = old_model.timestamp_id
            model_id = old_model.model_id
            if not metadata:
                metadata = old_model.get_metadata() # only read the old metadata if no metadata was provided
            existing_model = True
        else:
            # this is a weights pickle
            print("Loading weights from pickle")
            with open(filename, 'rb') as f:
                weights = dill.load(f)
    if not existing_model:
        model_id = model_id
        timestamp = 1610001000
        model = model_create_function()
        if not weights:
            if _is_tf_model(model):
                model.load_weights(default_weights_path)
                weights = model.get_weights()
            elif _is_tf_ensemble_model(model):
                # model is Ensemble tf
                weights=[]
                for ii in range(len(model)):
                    model[ii].load_weights(os.path.join(default_weights_path,f'fold_{ii}', 'best_metric_model.h5'))
                    weight = model[ii].get_weights()
                    weights.append(weight)
            elif _is_torch_model(model):
                import torch
                from dafne_dl.misc import torch_apply_fn_to_state_1
                model.load_state_dict(torch.load(default_weights_path, weights_only=True, map_location=torch.device('cpu')))
                weights = torch_apply_fn_to_state_1(model.state_dict(), lambda x: x.clone())
            elif _is_torch_ensemble_model(model):
                # model is Ensemble pytorch
                import torch
                from dafne_dl.misc import torch_apply_fn_to_state_1
                weights=[]
                for ii in range(len(model)):
                    model[ii].load_state_dict(torch.load(os.path.join(default_weights_path,f'fold_{ii}', 'best_metric_model.pt'), weights_only=True, map_location=torch.device('cpu')))
                    weight = torch_apply_fn_to_state_1(model[ii].state_dict(), lambda x: x.clone())
                    weights.append(weight)

        filename = os.path.join(base_folder, f'{model_name_prefix}_{timestamp}.model')

    modelObject = model_type(model_id,
                                 model_create_function,
                                 model_apply_function,
                                 data_preprocess_function=data_preprocess_function,  # function that preprocesses the data
                                 data_postprocess_function=data_postprocess_function,  # function that postprocesses the data
                                 label_preprocess_function=label_preprocess_function,  # function that preprocesses the labels for training
                                 incremental_learn_function=model_learn_function,
                                 weights=weights,
                                 timestamp_id=timestamp,
                                 data_dimensionality=dimensionality,
                                 metadata=metadata
                                 )

    with open(filename, 'wb') as f:
        modelObject.dump(f)

    try:
        os.remove(filename + '.sha256')
    except FileNotFoundError:
        pass

    print('Saved', filename)

    json_file_name = os.path.join(base_folder, f'{model_name_prefix}.json')
    with open(json_file_name, 'w') as f:
        modelObject.save_json_metadata(f, pretty=True)

def save_weights_torch(model_id,
                       model_path,
                       final_weights_path,
                       model_type=DynamicDLModel):
    """
    Function that extracts and save the weigths of an existing model.
    """
    import torch

    print("Converting model", model_path)
    old_model_path = model_path
    old_model = model_type.Load(open(old_model_path, 'rb'))
    shutil.copyfile(old_model_path, old_model_path + '.bak')
    weights = old_model.get_weights()
    save_model="best_metric_model.pt"
    timestamp = old_model.timestamp_id
    model_id = old_model.model_id

    for fold in range(len(weights)):
        save_path=os.path.join(final_weights_path,'{m}_{t}'.format(m=model_id, t=timestamp), 'fold_{f}'.format(f=fold))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(weights[fold], os.path.join(save_path, save_model))
    
    print('Saved Weights')

