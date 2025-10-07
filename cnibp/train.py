import argparse, os, time, torch, numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import TrainConfig
from .datasets import PPGWindowsDataset
from .losses import WeightedSmoothL1
from .metrics import mae, rmse, corr, aami, bhs_grades
from .utils import set_seed, ensure_dir, load_pid_list, save_json
from .models import PPGTransformerRegressor

def train_one_fold(args, fold, train_pids, test_pids, device):
    cfg = TrainConfig()
    # datasets
    tr_ds = PPGWindowsDataset(args.data_root, train_pids, use_sqi=args.use_sqi,
                              sqi_thresh=args.sqi_thresh, fs_default=cfg.fs_default)
    te_ds = PPGWindowsDataset(args.data_root, test_pids, use_sqi=False,
                              sqi_thresh=0.0, fs_default=cfg.fs_default)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # model + opt
    model = PPGTransformerRegressor(
        conv_channels=cfg.conv_channels,
        d_model=cfg.transformer_dim,
        nhead=cfg.transformer_heads,
        nlayers=cfg.transformer_layers,
        dropout=cfg.dropout
    ).to(device)

    loss_fn = WeightedSmoothL1(cfg.sbp_weight, cfg.dbp_weight, beta=1.0)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=args.epochs)
    best_mae = 1e9
    patience = cfg.patience
    out_fold = os.path.join(args.out_dir, f"fold{fold}")
    ensure_dir(out_fold)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, meta, y in tqdm(tr_dl, desc=f"Fold {fold} | Epoch {epoch+1}/{args.epochs}"):
            x, meta, y = x.to(device), meta.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x, meta)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sch.step()

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, meta, y in te_dl:
                x, meta = x.to(device), meta.to(device)
                pred = model(x, meta).cpu().numpy()
                y_pred.append(pred)
                y_true.append(y.numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        sbp_mae = mae(y_true[:,0], y_pred[:,0])
        dbp_mae = mae(y_true[:,1], y_pred[:,1])
        val_mae = 0.7*sbp_mae + 0.3*dbp_mae  # monitor metric prioritizing SBP

        stats = {
            "epoch": epoch+1,
            "sbp_mae": sbp_mae, "dbp_mae": dbp_mae,
            "sbp_rmse": rmse(y_true[:,0], y_pred[:,0]),
            "dbp_rmse": rmse(y_true[:,1], y_pred[:,1]),
            "sbp_r": corr(y_true[:,0], y_pred[:,0]),
            "dbp_r": corr(y_true[:,1], y_pred[:,1]),
            "sbp_aami": aami(y_true[:,0], y_pred[:,0]),
            "dbp_aami": aami(y_true[:,1], y_pred[:,1]),
            "sbp_bhs": bhs_grades(y_true[:,0], y_pred[:,0]),
            "dbp_bhs": bhs_grades(y_true[:,1], y_pred[:,1])
        }
        save_json(os.path.join(out_fold, "last_val.json"), stats)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join(out_fold, "best.pt"))
            patience = cfg.patience
            save_json(os.path.join(out_fold, "best_val.json"), stats)
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stop at epoch {epoch+1}")
                break

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    for fold in range(args.n_splits):
        train_pids = load_pid_list(os.path.join(args.splits_dir, f"fold{fold}_train.txt"))
        test_pids  = load_pid_list(os.path.join(args.splits_dir, f"fold{fold}_test.txt"))
        train_one_fold(args, fold, train_pids, test_pids, device)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_sqi", action="store_true")
    ap.add_argument("--sqi_thresh", type=float, default=0.8)
    main(ap.parse_args())
