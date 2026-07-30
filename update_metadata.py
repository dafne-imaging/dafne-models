import json
import sys
import shutil
import os

from dafne_dl.model_loaders import generic_load_model
from dafne_models.common import set_default_value

model_path = sys.argv[1]

dirname, basename = os.path.split(model_path)
model_name_prefix = os.path.basename(basename).split('_')[0]
json_path = os.path.join(dirname, model_name_prefix + '.json')
print(json_path)
print(model_name_prefix)

with open(model_path, 'rb') as model_file:
    model = generic_load_model(model_file)
with open(json_path) as json_file:
    metadata_json = json.load(json_file)

metadata = model.get_metadata()
metadata |= metadata_json
set_default_value(metadata, 'model_name', model_name_prefix)
set_default_value(metadata, 'model_type', type(model).__name__)
set_default_value(metadata, 'variants', [''])
set_default_value(metadata, 'categories', [])
set_default_value(metadata, 'dependencies', {})
model.set_metadata(metadata)
shutil.move(model_path, model_path + '.bak')
shutil.move(json_path, json_path + '.bak')
with open(json_path, 'w') as outfile:
    model.save_json_metadata(outfile, pretty=True)
with open(model_path, 'wb') as outfile:
    model.dump(outfile)