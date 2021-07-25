import json 
from types import SimpleNamespace
import os 

def write_default_config():
    default_config = {
        "model": "ViT-B/16",
        "batch_size": 32,
        "checkpoint_interval": 64,
        "top_k" : 5
    }
    with open("config/config.json", "w") as default_file:
        default_file.write(json.dumps(default_config))

config_file_name = "config/config.json"
env = {}
env_dict = {}

if not os.path.exists(config_file_name):
    write_default_config()

with open(config_file_name) as config_file:
    env_dict = json.loads(config_file.read())
    config_file.seek(0)
    config_obj = json.loads(config_file.read() , object_hook= lambda d: SimpleNamespace(**d))
    env = config_obj


