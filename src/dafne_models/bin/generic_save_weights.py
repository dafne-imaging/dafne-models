import sys
import dill
from dafne_dl.model_loaders import generic_load_model


def main():
    model_file = sys.argv[1]
    if len(sys.argv) > 2:
        weights_file = sys.argv[2]
    else:
        weights_file = model_file.replace('.model', '_weights.pickle')
    with open(model_file, 'rb') as f:
        m = generic_load_model(f)
    weights = m.get_weights()
    with open(weights_file, 'wb') as f:
        dill.dump(weights, f)

