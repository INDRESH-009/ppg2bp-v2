import numpy as np
from scipy.stats import pearsonr

def mae(y, yhat): return float(np.mean(np.abs(yhat - y)))
def rmse(y, yhat): return float(np.sqrt(np.mean((yhat - y)**2)))
def corr(y, yhat):
    if np.std(y) < 1e-6 or np.std(yhat) < 1e-6: return 0.0
    r, _ = pearsonr(y, yhat)
    return float(r)

def aami(y, yhat):
    diff = yhat - y
    return float(np.mean(diff)), float(np.std(diff))

def bhs_grades(y, yhat):
    abs_e = np.abs(yhat - y)
    pct5 = 100.0 * np.mean(abs_e <= 5.0)
    pct10 = 100.0 * np.mean(abs_e <= 10.0)
    pct15 = 100.0 * np.mean(abs_e <= 15.0)
    grade = "A" if (pct5>=60 and pct10>=85 and pct15>=95) else \
            "B" if (pct5>=50 and pct10>=75 and pct15>=90) else \
            "C" if (pct5>=40 and pct10>=65 and pct15>=85) else "D"
    return {"<=5": pct5, "<=10": pct10, "<=15": pct15, "grade": grade}
