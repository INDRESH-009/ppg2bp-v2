import os, glob, numpy as np
import torch
from torch.utils.data import Dataset
from .sqi import signal_quality_index

class PPGWindowsDataset(Dataset):
    """
    Expects data_root/<pid>/win*.npz with keys: ppg, sbp, dbp, age, gender, fs
    Filters windows by SQI if enabled.
    Returns (ppg_tensor[1,T], meta_tensor[2], target_tensor[2])
    meta = [age_norm, gender_binary], targets = [SBP, DBP] in mmHg
    """
    def __init__(self, data_root, pid_list, use_sqi=True, sqi_thresh=0.8, fs_default=125):
        self.samples = []
        self.use_sqi = use_sqi
        self.sqi_thresh = sqi_thresh
        self.fs_default = fs_default
        for pid in pid_list:
            folder = os.path.join(data_root, str(pid))
            files = sorted(glob.glob(os.path.join(folder, "win*_proc.npz")))
            for f in files:
                self.samples.append((pid, f))
        # lazy SQI (computed on __getitem__) to keep init fast

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, path = self.samples[idx]
        d = np.load(path)
        ppg = d["ppg"].astype(np.float32)
        sbp = float(d["sbp"])
        dbp = float(d["dbp"])
        age = float(d["age"])
        gender = int(d["gender"])  # assume 0/1
        fs = int(d["fs"]) if "fs" in d else self.fs_default

        # SQI gating
        if self.use_sqi:
            sqi = signal_quality_index(ppg, fs)
            if sqi < self.sqi_thresh:
                # resample a nearby index deterministically (wrap)
                j = (idx + 1) % len(self.samples)
                return self.__getitem__(j)

        # per-window z-normalize
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)

        # meta normalization (rough scale)
        age_norm = (age - 50.0) / 20.0
        gender_bin = 1.0 if gender == 1 else 0.0

        # to tensors
        ppg_t = torch.from_numpy(ppg).unsqueeze(0)  # [1,T]
        meta_t = torch.tensor([age_norm, gender_bin], dtype=torch.float32)  # [2]
        y_t = torch.tensor([sbp, dbp], dtype=torch.float32)  # [2]
        return ppg_t, meta_t, y_t
