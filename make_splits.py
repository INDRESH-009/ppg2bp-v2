import argparse, os, numpy as np
from sklearn.model_selection import KFold

def main(args):
    pids = sorted([d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d))])
    pids = np.array(pids)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    for fold, (tr, te) in enumerate(kf.split(pids)):
        np.savetxt(os.path.join(args.out_dir, f"fold{fold}_train.txt"), pids[tr], fmt="%s")
        np.savetxt(os.path.join(args.out_dir, f"fold{fold}_test.txt"), pids[te], fmt="%s")
    print(f"Saved splits in {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
