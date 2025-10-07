import argparse, os, glob, numpy as np, torch
from .models import PPGTransformerRegressor

def load_npz_from_input(inp):
    if os.path.isdir(inp):
        files = sorted(glob.glob(os.path.join(inp, "win*_proc.npz")))
        return files
    if os.path.isfile(inp) and inp.endswith(".npz"):
        return [inp]
    raise ValueError("Input must be a subject folder or a .npz file")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPGTransformerRegressor().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    files = load_npz_from_input(args.input)
    preds, gts = [], []

    with torch.no_grad():
        for f in files:
            d = np.load(f)
            ppg = d["ppg"].astype(np.float32)
            ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
            age = float(d["age"])
            gender = int(d["gender"])
            meta = np.array([(age - 50.0)/20.0, 1.0 if gender==1 else 0.0], dtype=np.float32)

            x = torch.from_numpy(ppg).unsqueeze(0).unsqueeze(0).to(device) # [1,1,T]
            m = torch.from_numpy(meta).unsqueeze(0).to(device)             # [1,2]
            yhat = model(x, m).cpu().numpy()[0]  # [2]
            preds.append(yhat)
            if "sbp" in d and "dbp" in d:
                gts.append([float(d["sbp"]), float(d["dbp"])])

    preds = np.array(preds)
    print("Predictions (SBP, DBP) first 10 windows:")
    print(preds[:10])
    if gts:
        gts = np.array(gts)
        mae = np.mean(np.abs(preds - gts), axis=0)
        print(f"MAE -> SBP: {mae[0]:.2f} | DBP: {mae[1]:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="Subject folder or single .npz")
    main(ap.parse_args())
