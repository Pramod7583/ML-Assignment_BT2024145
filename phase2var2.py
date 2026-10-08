import sys, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")

ROLL, VAR = "BT2024145", 2
DATA = sys.argv[1] if len(sys.argv) > 1 else "."
DEGREES = range(1, 14 + 1)
ALPHAS = [1e-6, 1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100]   # 1e-6 ~ plain polynomial


def make_model(deg, alpha):
    return make_pipeline(PolynomialFeatures(deg, include_bias=False),
                         StandardScaler(), Ridge(alpha=alpha))


train = pd.read_csv(f"{DATA}/{ROLL}_train_var{VAR}.csv")
test = pd.read_csv(f"{DATA}/{ROLL}_test_var{VAR}.csv")
X, y = train.drop(columns="y").values, train.y.values

# 1) score every (degree, alpha) with repeated 5-fold CV
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
rows = [(d, a, -cross_val_score(make_model(d, a), X, y, cv=cv,
                                scoring="neg_mean_squared_error").mean())
        for d in DEGREES for a in ALPHAS]
res = pd.DataFrame(rows, columns=["degree", "alpha", "cv_mse"])
res.to_csv(f"cv_results_var{VAR}.csv", index=False)

# 2) pick the best pair (and the best plain-polynomial degree for reference)
best = res.loc[res.cv_mse.idxmin()]
deg, alpha = int(best.degree), float(best.alpha)
plain = res[res.alpha == ALPHAS[0]]
print(f"CV MSE by degree (best alpha): {res.groupby('degree').cv_mse.min().round(3).to_dict()}")
print(f"Chosen: degree={deg}, alpha={alpha}, CV MSE={best.cv_mse:.4f}")
print(f"Best plain polynomial degree: {int(plain.loc[plain.cv_mse.idxmin()].degree)}")

# 3) 80/20 hold-out sanity check
idx = np.random.RandomState(0).permutation(len(X)); k = int(0.8 * len(X))
p = make_model(deg, alpha).fit(X[idx[:k]], y[idx[:k]]).predict(X[idx[k:]])
print(f"Hold-out MSE={mean_squared_error(y[idx[k:]], p):.4f}  R2={r2_score(y[idx[k:]], p):.4f}")

# 4) refit on all training data and predict the test file
model = make_model(deg, alpha).fit(X, y)
out = pd.DataFrame({"y": model.predict(test.values)})
out.to_csv(f"{ROLL}_pred_var{VAR}.csv", index=False)
print(f"Saved {ROLL}_pred_var{VAR}.csv ({len(out)} rows)")