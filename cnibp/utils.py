import os, json, torch, random, numpy as np

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def save_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def load_pid_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def ensure_dir(path): os.makedirs(path, exist_ok=True)
