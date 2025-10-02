# Sales Forecasting Description

A simple beginner-friendly sales forecasting workflow built inside the notebook `sales_forecasting_description.ipynb` using the provided CSV datasets.

## Datasets Used
- `train.csv`: Historical weekly sales (used to aggregate total weekly sales across stores/departments)
- `features.csv`: Additional time-based and economic indicators (merged by Date)
- `stores.csv`: Store metadata (loaded but not deeply used in this simple version)
- `test.csv`: Not used directly (focus is on building a forecasting approach from historical data)

## Steps Implemented in the Notebook
1. Imports: Basic data, plotting, and modeling libraries.
2. Data Loading: Reads `train.csv`, `features.csv`, `stores.csv` (parsing Date columns where needed).
3. Aggregation: Sums `Weekly_Sales` by `Date` to get overall weekly total sales.
4. Feature Aggregation: Averages numeric feature columns by `Date` and keeps holiday indicator (max).
5. Merge: Combines aggregated sales and features into a single time-indexed dataset.
6. Time Features: Adds `year`, `month`, `week`, `dayofweek`.
7. Lag Features: Creates `lag_1`, `lag_2`, `lag_4` of total weekly sales.
8. Dataset Prep: Drops rows with missing lag values and selects modeling columns.
9. Temporal Split: Uses the first 80% of rows as training, last 20% as a hold-out test (no shuffling).
10. Models Trained: Linear Regression and Random Forest Regressor.
11. Evaluation: Computes MAE and RMSE on the test split; selects the better model.
12. Actual vs Predicted Plot: Visual comparison of predictions vs ground truth.
13. Sales Trend Plot: Line chart of total weekly sales over the whole time range.
14. Correlation Heatmap: Correlation matrix of features plus target.
15. Residual Analysis: Residual time series and distribution histogram.
16. Feature Importance: Random Forest feature importance bar chart.
17. Next-Period Forecast: Single next-week prediction using the best model and lagged features.
18. Summary: Key metrics and forecast printed.
19. Rolling Averages: Added 4, 8, and 12-period rolling mean features for enhanced temporal smoothing.
20. Seasonal Decomposition: Used `statsmodels` additive decomposition (period=52) to inspect trend/seasonality/residual components.
21. Time-Aware Validation (XGBoost): Applied expanding window `TimeSeriesSplit` (5 folds) with XGBoost using lag, calendar, exogenous, and rolling features.
22. Final XGBoost Evaluation: Trained on full training portion, evaluated on hold-out, compared RMSE improvement over previous best model.

## Models
- Linear Regression (baseline)
- Random Forest Regressor (ensemble)
- XGBoost Regressor (gradient boosting, with rolling features and time-series CV)

Metrics reported:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## Forecasting Approach
A single-step (next period) forecast using lagged sales, calendar features, and available numeric exogenous variables. Advanced section enhances feature richness with rolling statistics and evaluates a boosted tree model under proper time-aware validation.

## Bonus Enhancements Added
- Rolling window means (4/8/12) to capture local level and smoothed trends.
- Seasonal decomposition for exploratory insight into yearly seasonality (weekly frequency approximated to 52 periods).
- Time-series cross-validation using `TimeSeriesSplit` to avoid lookahead bias.
- XGBoost model leveraging engineered features and compared against earlier best model.

## How to Run
1. Ensure Python 3.9+ (or similar) is installed.
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook sales_forecasting_description.ipynb
```
4. Run all cells in order.

## Possible Next Improvements
- Add rolling standard deviation and momentum (difference) features.
- Introduce holiday distance features (days until next holiday).
- Multi-step recursive or direct forecasting horizon extension.
- Hyperparameter tuning (Optuna / grid search) for XGBoost.
- Try LightGBM or CatBoost for potentially faster/better performance.
- Store/department hierarchical reconciliation instead of total aggregation.

## Requirements
See `requirements.txt` (now includes `statsmodels` and `xgboost` for decomposition and boosted modeling).

## License
See `LICENSE` file if provided.
