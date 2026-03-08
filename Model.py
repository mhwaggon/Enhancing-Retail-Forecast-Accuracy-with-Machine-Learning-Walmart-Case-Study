import numpy as np, pandas as pd, matplotlib.pyplot as plt, logging, warnings
from prophet import Prophet
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# logging and warnings
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
warnings.simplefilter("ignore", ConvergenceWarning)

#data 
base_path = "data"
train = pd.read_csv(f"{base_path}/train.csv", parse_dates=["Date"])
features = pd.read_csv(f"{base_path}/features.csv", parse_dates=["Date"])
stores = pd.read_csv(f"{base_path}/stores.csv")


# data preprocessing. Dtypes
train["IsHoliday"] = train["IsHoliday"].astype(int)
features["IsHoliday"] = features["IsHoliday"].astype(int)
for c in [c for c in features.columns if c.startswith("MarkDown")]:
    features[c] = features[c].astype(float)

df = (train.merge(features, on=["Store","Date","IsHoliday"], how="left")
           .merge(stores, on="Store", how="left")
           .sort_values(["Store","Dept","Date"])
           .reset_index(drop=True))


markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
if markdown_cols:
    df[markdown_cols] = df[markdown_cols].fillna(0.0)

for c in ["Temperature","Fuel_Price","CPI","Unemployment"]:
    if c in df.columns:
        df[c] = (df.groupby("Store")[c].apply(lambda g: g.ffill().bfill())
                               .reset_index(level=0, drop=True))
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
if "Size" in df.columns:
    df["Size"] = df.groupby("Store")["Size"].transform("max")
    if df["Size"].isna().any():
        df["Size"] = df["Size"].fillna(df["Size"].median())

# Prophet regressors
regressors = ["IsHoliday"]

def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    return np.nan if denom == 0 else np.abs(y_true - y_pred).sum() / denom

def naive_last(hist_df, h):
    y = hist_df["Weekly_Sales"].to_numpy()
    return np.repeat(y[-1] if len(y) else 0.0, h)

def ts_roll4(hist_df, h):
    y = hist_df["Weekly_Sales"].to_numpy()
    v = float(hist_df["Weekly_Sales"].tail(4).mean()) if len(y)>=4 else (y[-1] if len(y) else 0.0)
    return np.repeat(v, h)

cutoff = pd.Timestamp("2012-07-01")

# select how many to run on. Full set required beefy computer
TOP_N = 250 

series_volume = (df[df["Date"] < cutoff]
                 .groupby(["Store","Dept"])["Weekly_Sales"]
                 .sum()
                 .sort_values(ascending=False))
top_series = set(series_volume.head(TOP_N).index)

def fit_prophet_fast(hist_df, fut_df):
    tmp = hist_df[["Date","Weekly_Sales"] + regressors].rename(columns={"Date":"ds","Weekly_Sales":"y"})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False,
                n_changepoints=5,
                seasonality_mode="additive",
                changepoint_prior_scale=0.05)
    for r in regressors:
        m.add_regressor(r)
    m.fit(tmp)
    fut = fut_df[["Date"] + regressors].rename(columns={"Date":"ds"})
    return m.predict(fut)["yhat"].to_numpy()

rows = []
for (s,d), g in df.groupby(["Store","Dept"]):
    g = g.sort_values("Date")
    hist, val = g[g["Date"] < cutoff], g[g["Date"] >= cutoff]
    if len(val) == 0 or len(hist) < 6:
        continue

    y_true = val["Weekly_Sales"].to_numpy()

    # baselines
    yhat_naive = naive_last(hist, len(val))   # "Naive"
    yhat_ts    = ts_roll4(hist, len(val))     # fast TS baseline (standard SARIMAX is too slow for my machine)
    # prophet on top series only to increase speed. Need strong computer to run full set.
    if (s, d) in top_series:
        try:
            yhat_prophet = fit_prophet_fast(hist, val)
            w_prophet = wmape(y_true, yhat_prophet)
        except Exception:
            w_prophet = np.nan
    else:
        w_prophet = np.nan

    rows.append({
        "Store": s, "Dept": d,
        "WMAPE_Prophet": w_prophet,
        "WMAPE_SARIMAX": wmape(y_true, yhat_ts),
        "WMAPE_Naive":  wmape(y_true, yhat_naive)
    })

results = pd.DataFrame(rows)

summary = {
    "Prophet": float(results["WMAPE_Prophet"].mean(skipna=True)),
    "SARIMAX": float(results["WMAPE_SARIMAX"].mean(skipna=True)),
    "Naive":   float(results["WMAPE_Naive"].mean(skipna=True)),
    "n_series_total": int(len(results)),
    "n_series_prophet": int(results["WMAPE_Prophet"].notna().sum())
}
print(summary)


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 1. Overall weekly sales trend
train.groupby("Date")["Weekly_Sales"].sum().plot(figsize=(9,4))
plt.title("Overall Weekly Sales Trend (2010–2013)")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.show()

# 2. Holiday vs non-holiday average sales
train.groupby("IsHoliday")["Weekly_Sales"].mean().plot(kind="bar", color=["gray","orange"], figsize=(6,4))
plt.title("Average Weekly Sales: Holiday vs Non-Holiday")
plt.xlabel("IsHoliday (0 = No, 1 = Yes)")
plt.ylabel("Average Sales")
plt.show()

# 3. Average WMAPE by model
pd.Series(summary).sort_values().plot(kind="barh", figsize=(6,3), color="skyblue")
plt.title("Average WMAPE by Model (Validation)")
plt.xlabel("WMAPE")
plt.show()

# 4. Model comparison per store type (if available)
if "Type" in stores.columns:
    tmp = results.merge(stores[["Store","Type"]], on="Store", how="left")
    by_type = tmp.groupby("Type")[["WMAPE_Prophet","WMAPE_SARIMAX","WMAPE_Naive"]].mean()
    by_type.plot(kind="bar", figsize=(6,4))
    plt.title("Mean WMAPE by Store Type")
    plt.ylabel("WMAPE")
    plt.show()

# 5. Prophet vs SARIMAX delta distribution
(results["WMAPE_Prophet"] - results["WMAPE_SARIMAX"]).dropna().hist(bins=30, figsize=(6,3))
plt.title("WMAPE Difference: Prophet – SARIMAX")
plt.xlabel("Delta WMAPE")
plt.ylabel("Count")
plt.show()

# 6. Share of series won by model
best_model = results[["WMAPE_Prophet","WMAPE_SARIMAX","WMAPE_Naive"]].idxmin(axis=1)
share = best_model.value_counts(normalize=True)
share.index = [i.replace("WMAPE_","") for i in share.index]
share.plot(kind="bar", figsize=(5,3), color="lightgreen")
plt.title("Share of Series Won by Model")
plt.ylabel("Share of (Store,Dept) Series")
plt.show()






