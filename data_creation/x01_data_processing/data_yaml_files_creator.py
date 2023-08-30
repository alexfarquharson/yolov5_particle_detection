import yaml
from data_configs import test_data_params, train_data_params, extra_test_data_params
import re

save_dir = "../../data/"
# Loop through each key-value pair in the dictionary and create a YAML file for each
for key, value in test_data_params.items():
    keyx = re.sub(r"\.", "_", key)
    keyx = re.sub(r"\%", "", keyx)
    keyx = re.sub(r" ", "_", keyx)
    file_path = save_dir + "dataset_" + keyx + "_test.yaml"
    with open(file_path, 'w') as yaml_file:
        yaml_data = {
            'path': "data_creation/" + value['subdir'][:-1],
            'train': 'images/test',
            'val': 'images/test',
            'test': 'images/test',
            'names': {
                0: 'Spot'
            }
        }
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
        

# Loop through each key-value pair in the dictionary and create a YAML file for each
for key, value in extra_test_data_params.items():
    keyx = re.sub(r"\.", "_", key)
    keyx = re.sub(r"\%", "", keyx)
    keyx = re.sub(r" ", "_", keyx)
    file_path = save_dir + "dataset_" + keyx + "_test.yaml"
    with open(file_path, 'w') as yaml_file:
        yaml_data = {
            'path': "data_creation/" + value['subdir'][:-1],
            'train': 'images/test',
            'val': 'images/test',
            'test': 'images/test',
            'names': {
                0: 'Spot'
            }
        }
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)


# Loop through each key-value pair in the dictionary and create a YAML file for each
for key, value in train_data_params.items():
    keyx = re.sub(r"\.", "_", key)
    keyx = re.sub(r"\%", "", keyx)
    keyx = re.sub(r" ", "_", keyx)
    file_path = save_dir + "dataset_" + keyx[:-5] + '.yaml'
    with open(file_path, 'w') as yaml_file:
        yaml_data = {
            'path': "data_creation/" + value['subdir'][:-1],
            'train': 'images/train',
            'val': 'images/valid',
            'test': '',
            'names': {
                0: 'Spot'
            }
        }
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
        