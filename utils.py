import os
import json
import torch # type: ignore
import numpy as np

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _config_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".config")

def _model_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".tar")

def load_model(model_path, model_name, act_fn_by_name, Network, net = None):
    config_file, model_file = _config_file(model_path, model_name), _model_file(model_path, model_name)
    
    if not os.path.isfile(config_file):
        raise Exception(f"Could not find the config file: {config_file}")
    
    if not os.path.isfile(model_file):
        raise Exception(f"Could not find the config file: {model_file}")
    
    with open(config_file) as f:
        config_dict = json.load(f)

    if net is None:
        act_fn_name = config_dict["act_fn"].pop("name").lower()
        act_fn = act_fn_by_name[act_fn_name](**config_dict.pop("act_fn"))
        net = Network(act_fn=act_fn, **config_dict)

    net.load_state_dict(torch.load(model_file))

    return net

def save_model(model, model_path, model_name):
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _config_file(model_path, model_name), _model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(model.config, f)
    torch.save(model.state_dict(), model_file)