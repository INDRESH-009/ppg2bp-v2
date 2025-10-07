import argparse, os, json, numpy as np

def main(args):
    folds = []
    for name in os.listdir(args.out_dir):
        if name.startswith("fold"):
            path = os.path.join(args.out_dir, name, "best_val.json")
            if os.path.exists(path):
                with open(path) as f: folds.append(json.load(f))
    if not folds:
        print("No fold results found.")
        return
    def avg_key(k, as_tuple=False):
        vals = []
        for f in folds:
            v = f[k]
            if as_tuple: vals.append(tuple(v))
            else: vals.append(v)
        if as_tuple: return tuple(np.mean(vals, axis=0).tolist())
        return float(np.mean(vals))

    summary = {
        "folds": len(folds),
        "sbp_mae": avg_key("sbp_mae"),
        "dbp_mae": avg_key("dbp_mae"),
        "sbp_rmse": avg_key("sbp_rmse"),
        "dbp_rmse": avg_key("dbp_rmse"),
        "sbp_r": avg_key("sbp_r"),
        "dbp_r": avg_key("dbp_r")
    }
    outp = os.path.join(args.out_dir, "cv_summary.json")
    with open(outp, "w") as f: json.dump(summary, f, indent=2)
    print("CV summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
