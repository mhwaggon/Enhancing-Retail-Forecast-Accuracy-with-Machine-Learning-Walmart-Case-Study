# Retail Forecasting

A Python-based retail forecasting project that predicts weekly sales at the store-department level using time series models. This project compares simple baselines, a rolling average time series approach, and Prophet to measure forecasting performance across many retail series.

## Overview

Retail forecasting helps businesses plan inventory, staffing, promotions, and supply chain decisions. In this project, weekly sales data is combined with store information and external features such as holidays, fuel prices, CPI, unemployment, temperature, and markdowns.

The goal is to evaluate how well different forecasting approaches perform across many store-department combinations.

The models compared are:

- Naive baseline
- Rolling average time series baseline
- Prophet with holiday regressor

Performance is measured using WMAPE, which is commonly used in forecasting because it scales errors by total sales volume.

## Dataset Files

This project expects the following CSV files inside a `data/` folder:

- `train.csv`
- `features.csv`
- `stores.csv`

## Expected Columns

### `train.csv`
- `Store`
- `Dept`
- `Date`
- `Weekly_Sales`
- `IsHoliday`

### `features.csv`
- `Store`
- `Date`
- `IsHoliday`
- `Temperature`
- `Fuel_Price`
- `CPI`
- `Unemployment`
- `MarkDown1` through `MarkDown5` (if available)

### `stores.csv`
- `Store`
- `Type`
- `Size`

## Project Workflow

### 1. Load and merge the data
The project loads training, feature, and store-level data, then merges them into a single modeling dataset using:

- `Store`
- `Date`
- `IsHoliday`

### 2. Preprocess the data
Several preprocessing steps are applied:

- Convert holiday indicators to integers
- Convert markdown columns to numeric
- Fill missing markdown values with `0`
- Forward-fill and backward-fill numeric feature columns by store
- Fill remaining missing values with medians
- Standardize store size within each store

### 3. Create store-department time series
Each unique `(Store, Dept)` pair is treated as its own sales time series.

### 4. Split into train and validation periods
The validation cutoff is:

- `2012-07-01`

Data before this date is used as historical training data, and data on or after this date is used for validation.

### 5. Train and evaluate forecasting models
For each store-department series, the project compares:

#### Naive baseline
Forecasts all future periods using the last observed weekly sales value.

#### Rolling average baseline
Forecasts using the average of the last 4 observed weekly sales values.

#### Prophet
Uses weekly and yearly seasonality with `IsHoliday` as a regressor.

To improve runtime, Prophet is only fit on the top sales-volume series.

### 6. Compare model accuracy
Each model is evaluated using WMAPE:

`WMAPE = sum(|actual - forecast|) / sum(|actual|)`

Lower WMAPE indicates better forecasting performance.

## Why WMAPE?

WMAPE is useful in retail forecasting because it:

- Handles scale differences better than raw error metrics
- Is easy to interpret
- Reflects business impact more clearly than many standard metrics

## Models Used

### 1. Naive Forecast
A simple benchmark that assumes the next value will be the same as the most recent observed value.

### 2. Rolling 4-Week Average
A lightweight time series baseline that uses recent sales history to smooth short-term fluctuations.

### 3. Prophet
A forecasting model designed to handle trend and seasonality. In this project, Prophet includes:

- Weekly seasonality
- Yearly seasonality
- Holiday regressor
- Limited changepoints for faster fitting

## Key Parameters

### Validation cutoff
- `2012-07-01`

### Prophet settings
- `weekly_seasonality=True`
- `yearly_seasonality=True`
- `daily_seasonality=False`
- `n_changepoints=5`
- `seasonality_mode="additive"`
- `changepoint_prior_scale=0.05`

### Runtime optimization
- `TOP_N = 250`

Prophet is only trained on the top 250 store-department series by total historical sales volume to keep runtime manageable on weaker machines.

## Output

The script produces:

### 1. Model performance summary
A dictionary showing average WMAPE for each model and how many series were evaluated.

Example structure:

```python
{
    "Prophet": 0.18,
    "SARIMAX": 0.21,
    "Naive": 0.27,
    "n_series_total": 500,
    "n_series_prophet": 250
}
