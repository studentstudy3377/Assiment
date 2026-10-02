"""Advanced forecasting, backtesting, uncertainty, anomaly detection, and exports."""
from __future__ import annotations

from io import BytesIO
import json
import math
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


EPS = 1e-9


def _trend_features(t):
    t = np.asarray(t, dtype=float)
    return np.column_stack([t])


def _seasonal_features(t):
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        t,
        np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12),
        np.sin(4 * np.pi * t / 12), np.cos(4 * np.pi * t / 12),
    ])


def _gbm_features(history, step_index):
    """Create one autoregressive feature row from history."""
    h = np.asarray(history, dtype=float)
    lags = [1, 2, 3, 6, 12]
    vals = [h[-lag] if len(h) >= lag else np.nan for lag in lags]
    recent3 = np.mean(h[-min(3, len(h)):])
    recent6 = np.mean(h[-min(6, len(h)):])
    trend3 = 0.0 if len(h) < 3 else h[-1] - h[-3]
    month_sin = math.sin(2 * math.pi * step_index / 12)
    month_cos = math.cos(2 * math.pi * step_index / 12)
    return np.array(vals + [recent3, recent6, trend3, month_sin, month_cos], dtype=float)


def _fit_gbm_and_forecast(y_train, horizon):
    y_train = np.asarray(y_train, dtype=float)
    rows, targets = [], []
    min_hist = 6
    for i in range(min_hist, len(y_train)):
        row = _gbm_features(y_train[:i], i)
        if np.isnan(row).any():
            # Replace unavailable long lags with current expanding mean.
            row = np.where(np.isnan(row), np.mean(y_train[:i]), row)
        rows.append(row)
        targets.append(y_train[i])
    if len(rows) < 8:
        raise ValueError('Not enough observations for autoregressive boosting.')
    model = GradientBoostingRegressor(
        n_estimators=120, learning_rate=0.035, max_depth=2,
        min_samples_leaf=2, loss='huber', random_state=42,
    )
    model.fit(np.vstack(rows), np.asarray(targets))
    hist = list(y_train)
    out = []
    for _ in range(int(horizon)):
        idx = len(hist)
        row = _gbm_features(hist, idx)
        row = np.where(np.isnan(row), np.mean(hist), row)
        pred = float(model.predict(row.reshape(1, -1))[0])
        out.append(pred)
        hist.append(pred)
    return np.asarray(out)


def _fit_predict(model_name, y_train, future_idx):
    y_train = np.asarray(y_train, dtype=float)
    n = len(y_train)
    future_idx = np.asarray(future_idx, dtype=int)
    horizon = len(future_idx)

    if model_name == 'Last value':
        return np.repeat(y_train[-1], horizon)
    if model_name == '3-month moving average':
        hist = list(y_train)
        out = []
        for _ in range(horizon):
            p = float(np.mean(hist[-min(3, len(hist)):]))
            out.append(p); hist.append(p)
        return np.asarray(out)
    if model_name == 'Seasonal naive (12M)':
        hist = list(y_train)
        out = []
        for _ in range(horizon):
            p = float(hist[-12]) if len(hist) >= 12 else float(hist[-1])
            out.append(p); hist.append(p)
        return np.asarray(out)
    if model_name == 'Linear trend':
        m = LinearRegression().fit(_trend_features(np.arange(n)), y_train)
        return m.predict(_trend_features(future_idx))
    if model_name == 'Trend + annual seasonality':
        m = LinearRegression().fit(_seasonal_features(np.arange(n)), y_train)
        return m.predict(_seasonal_features(future_idx))
    if model_name == 'Holt trend':
        if not HAS_STATSMODELS:
            raise ValueError('statsmodels is unavailable.')
        fit = ExponentialSmoothing(y_train, trend='add', seasonal=None, initialization_method='estimated').fit(optimized=True)
        return np.asarray(fit.forecast(horizon), dtype=float)
    if model_name == 'Holt-Winters seasonal':
        if not HAS_STATSMODELS:
            raise ValueError('statsmodels is unavailable.')
        fit = ExponentialSmoothing(
            y_train, trend='add', seasonal='add', seasonal_periods=12,
            damped_trend=True, initialization_method='estimated'
        ).fit(optimized=True, use_brute=False)
        return np.asarray(fit.forecast(horizon), dtype=float)
    if model_name == 'Autoregressive Gradient Boosting':
        return _fit_gbm_and_forecast(y_train, horizon)
    raise ValueError(f'Unsupported model: {model_name}')


def _eligible_models(n_train):
    models = ['Last value', '3-month moving average']
    if n_train >= 7:
        models += ['Linear trend']
        if HAS_STATSMODELS:
            models += ['Holt trend']
    if n_train >= 14:
        models += ['Seasonal naive (12M)', 'Trend + annual seasonality']
    if n_train >= 20:
        models += ['Autoregressive Gradient Boosting']
    if n_train >= 24 and HAS_STATSMODELS:
        models += ['Holt-Winters seasonal']
    return models


def _metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = np.abs(actual) > EPS
    mape = float(np.mean(np.abs(err[nz]) / np.abs(actual[nz])) * 100) if nz.any() else None
    denom = np.abs(actual) + np.abs(pred)
    ok = denom > EPS
    smape = float(np.mean(2 * np.abs(err[ok]) / denom[ok]) * 100) if ok.any() else None
    wmape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(actual)), EPS) * 100)
    bias = float(np.mean(err))
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape, 'wmape': wmape, 'bias': bias}


def _rolling_backtest(y, model_name, max_folds=8):
    y = np.asarray(y, dtype=float)
    min_train = 6
    if model_name in {'Seasonal naive (12M)', 'Trend + annual seasonality'}:
        min_train = 14
    elif model_name == 'Autoregressive Gradient Boosting':
        min_train = 20
    elif model_name == 'Holt-Winters seasonal':
        min_train = 24
    starts = list(range(max(min_train, len(y) - max_folds), len(y)))
    rows = []
    for origin in starts:
        if origin >= len(y):
            continue
        train = y[:origin]
        try:
            pred = float(_fit_predict(model_name, train, [origin])[0])
        except Exception:
            continue
        actual = float(y[origin])
        rows.append({'Origin': origin, 'Actual': actual, 'Prediction': pred, 'Error': pred - actual})
    return pd.DataFrame(rows)


def _build_model_comparison(y):
    models = _eligible_models(len(y))
    summaries, residuals = [], {}
    for name in models:
        bt = _rolling_backtest(y, name)
        if len(bt) < 2:
            continue
        met = _metrics(bt['Actual'], bt['Prediction'])
        residuals[name] = bt['Error'].to_numpy(dtype=float)
        summaries.append({
            'Model': name, 'CV folds': len(bt), 'MAE': met['mae'], 'RMSE': met['rmse'],
            'MAPE (%)': met['mape'], 'sMAPE (%)': met['smape'], 'wMAPE (%)': met['wmape'],
            'Bias': met['bias'],
        })
    if not summaries:
        raise ValueError('Could not complete rolling backtests on the selected history.')
    comp = pd.DataFrame(summaries).sort_values(['MAE', 'RMSE']).reset_index(drop=True)
    return comp, residuals


def _forecast_ensemble(y, horizon, model_comparison):
    future_idx = np.arange(len(y), len(y) + horizon)
    top = model_comparison.head(min(3, len(model_comparison))).copy()
    preds, names, weights = [], [], []
    for _, r in top.iterrows():
        name = str(r['Model'])
        try:
            p = _fit_predict(name, y, future_idx)
        except Exception:
            continue
        preds.append(np.asarray(p, dtype=float)); names.append(name)
        weights.append(1.0 / max(float(r['MAE']), EPS))
    if not preds:
        raise ValueError('No candidate model could generate the final forecast.')
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    matrix = np.vstack(preds)
    ensemble = np.average(matrix, axis=0, weights=weights)
    members = pd.DataFrame({'Model': names, 'Weight': weights})
    return ensemble, members, matrix


def _forecast_intervals(point_forecast, residual_pool, recent_volatility):
    residual_pool = np.asarray(residual_pool, dtype=float)
    if len(residual_pool) >= 4:
        q10, q90 = np.quantile(residual_pool, [0.10, 0.90])
        q025, q975 = np.quantile(residual_pool, [0.025, 0.975])
        spread80 = max(abs(q10), abs(q90), recent_volatility * 0.25)
        spread95 = max(abs(q025), abs(q975), recent_volatility * 0.45, spread80 * 1.35)
    else:
        base = max(float(np.std(residual_pool)) if len(residual_pool) else 0.0, recent_volatility * 0.35)
        spread80, spread95 = 1.28 * base, 1.96 * base
    scale = np.sqrt(np.arange(1, len(point_forecast) + 1))
    p10 = point_forecast - spread80 * scale
    p90 = point_forecast + spread80 * scale
    p025 = point_forecast - spread95 * scale
    p975 = point_forecast + spread95 * scale
    return p10, p90, p025, p975


def _detect_anomalies(monthly):
    values = monthly.to_numpy(dtype=float)
    n = len(values)
    if n < 6:
        return pd.DataFrame(columns=['Month', 'Revenue', 'Expected', 'Residual', 'Robust z-score', 'Severity'])

    t = np.arange(n)
    # Remove broad trend and, where possible, annual seasonality before judging a month as unusual.
    if n >= 14:
        model = LinearRegression().fit(_seasonal_features(t), values)
        expected = model.predict(_seasonal_features(t))
    else:
        model = LinearRegression().fit(_trend_features(t), values)
        expected = model.predict(_trend_features(t))

    residual = values - expected
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    robust_sigma = 1.4826 * mad
    classical_sigma = float(np.std(residual, ddof=1)) if n > 1 else 0.0
    sigma = max(robust_sigma, classical_sigma * 0.60, EPS)
    z = (residual - med) / sigma
    abs_z = np.abs(z)
    keep = abs_z >= 3.0
    severity = np.where(abs_z >= 4.5, 'Critical', np.where(abs_z >= 3.5, 'High', 'Watch'))
    out = pd.DataFrame({
        'Month': monthly.index.astype(str), 'Revenue': values, 'Expected': expected,
        'Residual': residual, 'Robust z-score': z, 'Severity': severity,
    })
    return out.loc[keep].reset_index(drop=True)


def analyze(frame, date_column, revenue_column, remove_duplicates=False, forecast_horizon=3):
    if date_column == revenue_column:
        raise ValueError('Choose different columns for the date and Net Revenue.')
    if not 1 <= int(forecast_horizon) <= 18:
        raise ValueError('Forecast horizon must be between 1 and 18 months.')

    duplicate_count = int(frame.duplicated().sum())
    data = frame.drop_duplicates().copy() if remove_duplicates else frame.copy()
    raw_dates = data[date_column]
    numeric_dates = raw_dates.map(lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)))
    dates = pd.to_datetime(raw_dates.where(~numeric_dates), format='mixed', errors='coerce')
    if numeric_dates.any():
        serials = pd.to_numeric(raw_dates.loc[numeric_dates], errors='coerce')
        if not serials.dropna().between(1, 100000).all():
            raise ValueError('Numeric dates must be Excel date serials. Choose a calendar date column.')
        dates.loc[numeric_dates] = pd.to_datetime(serials, unit='D', origin='1899-12-30', errors='coerce')

    revenue = pd.to_numeric(data[revenue_column], errors='coerce')
    bad_dates = dates.isna()
    bad_revenue = revenue.isna() | ~np.isfinite(revenue)
    if bad_dates.any() or bad_revenue.any():
        raise ValueError(f'{int(bad_dates.sum())} invalid dates and {int(bad_revenue.sum())} invalid revenue values. Correct those rows or select different fields.')

    periods = dates.dt.to_period('M')
    monthly = revenue.groupby(periods).sum().sort_index()
    if len(monthly) < 6:
        raise ValueError('At least 6 monthly observations are required for forecasting and backtesting.')
    missing = pd.period_range(monthly.index.min(), monthly.index.max(), freq='M').difference(monthly.index)
    if len(missing):
        raise ValueError('Missing months: ' + ', '.join(missing.astype(str)) + '. Add those months so the time series is continuous.')

    y = monthly.to_numpy(dtype=float)
    model_comparison, residual_map = _build_model_comparison(y)
    best_model = str(model_comparison.iloc[0]['Model'])
    forecast_values, ensemble_members, member_matrix = _forecast_ensemble(y, int(forecast_horizon), model_comparison)

    # Use the selected model's one-step rolling backtest as a transparent validation view.
    best_bt = _rolling_backtest(y, best_model)
    holdout_n = min(3, len(best_bt))
    holdout_bt = best_bt.tail(holdout_n).copy()
    holdout_metrics = _metrics(holdout_bt['Actual'], holdout_bt['Prediction'])

    # Pool residuals from top models, giving a more robust empirical uncertainty estimate.
    residual_pool = []
    for name in ensemble_members['Model']:
        residual_pool.extend(residual_map.get(name, []))
    recent_volatility = float(np.std(y[-min(6, len(y)):]))
    p10, p90, p025, p975 = _forecast_intervals(forecast_values, residual_pool, recent_volatility)

    future = pd.period_range(monthly.index[-1] + 1, periods=int(forecast_horizon), freq='M')
    history = pd.DataFrame({
        'Month': monthly.index.astype(str), 'Actual Net Revenue': y,
        'MoM Growth (%)': monthly.pct_change().to_numpy() * 100,
        'Rolling 3M Avg': monthly.rolling(3, min_periods=1).mean().to_numpy(),
        'Rolling 6M Avg': monthly.rolling(6, min_periods=1).mean().to_numpy(),
        'Source rows': data.groupby(periods).size().reindex(monthly.index).to_numpy(),
    })
    holdout = pd.DataFrame({
        'Month': monthly.index[holdout_bt['Origin'].astype(int)].astype(str) if len(holdout_bt) else [],
        'Actual Net Revenue': holdout_bt['Actual'].to_numpy() if len(holdout_bt) else [],
        'Hold-out prediction': holdout_bt['Prediction'].to_numpy() if len(holdout_bt) else [],
        'Absolute error': np.abs(holdout_bt['Error'].to_numpy()) if len(holdout_bt) else [],
    })
    forecast = pd.DataFrame({
        'Month': future.astype(str), 'Forecast Net Revenue': forecast_values,
        'P10': p10, 'P90': p90, 'P2.5': p025, 'P97.5': p975,
    })

    mom = monthly.pct_change()
    latest_growth = None if mom.dropna().empty else float(mom.iloc[-1] * 100)
    avg_3m = float(monthly.iloc[-min(3, len(monthly)):].mean())
    prev_3m = float(monthly.iloc[-6:-3].mean()) if len(monthly) >= 6 else np.nan
    three_month_momentum = None if not np.isfinite(prev_3m) or abs(prev_3m) <= EPS else (avg_3m / prev_3m - 1) * 100
    yoy = None
    if len(monthly) >= 13 and abs(monthly.iloc[-13]) > EPS:
        yoy = float((monthly.iloc[-1] / monthly.iloc[-13] - 1) * 100)

    trend_slope = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) >= 2 else 0.0
    cv_best = model_comparison.iloc[0]
    forecast_cv = float(cv_best['RMSE'] / max(abs(np.mean(y[-min(6, len(y)):])), EPS) * 100)
    risk_level = 'Low' if forecast_cv < 10 else ('Moderate' if forecast_cv < 25 else 'High')

    anomalies = _detect_anomalies(monthly)
    summary = {
        'date_column': str(date_column), 'revenue_column': str(revenue_column),
        'source_rows': len(frame), 'rows_used': len(data), 'exact_duplicate_rows': duplicate_count,
        'duplicate_policy': 'removed' if remove_duplicates else 'retained',
        'historical_start': str(monthly.index[0]), 'historical_end': str(monthly.index[-1]),
        'history_months': len(monthly), 'currency': 'Workbook units; currency unspecified',
        'model': best_model, 'forecast_method': 'Weighted ensemble of top rolling-CV models',
        'forecast_horizon_months': int(forecast_horizon), 'holdout_months': int(holdout_n),
        'holdout_mape_pct': holdout_metrics['mape'], 'holdout_smape_pct': holdout_metrics['smape'],
        'holdout_wmape_pct': holdout_metrics['wmape'], 'holdout_mae': holdout_metrics['mae'],
        'holdout_rmse': holdout_metrics['rmse'], 'holdout_bias': holdout_metrics['bias'],
        'cv_best_mae': float(cv_best['MAE']), 'cv_best_rmse': float(cv_best['RMSE']),
        'cv_best_wmape_pct': float(cv_best['wMAPE (%)']), 'forecast_risk': risk_level,
        'latest_month_revenue': float(monthly.iloc[-1]), 'latest_mom_growth_pct': latest_growth,
        'latest_yoy_growth_pct': yoy, 'three_month_avg': avg_3m,
        'three_month_momentum_pct': three_month_momentum, 'monthly_trend_slope': trend_slope,
        'forecast_total': float(np.sum(forecast_values)), 'forecast_avg_month': float(np.mean(forecast_values)),
        'anomaly_count': int(len(anomalies)), 'statsmodels_available': HAS_STATSMODELS,
    }
    return {
        'monthly': monthly, 'history': history, 'holdout': holdout, 'forecast': forecast,
        'model_comparison': model_comparison, 'ensemble_members': ensemble_members,
        'anomalies': anomalies, 'summary': summary,
    }


def simulate_forecast(result, iterations=5000, seed=42):
    """Monte Carlo paths calibrated to the forecast's empirical uncertainty bands."""
    rng = np.random.default_rng(seed)
    f = result['forecast']
    mean = f['Forecast Net Revenue'].to_numpy(dtype=float)
    sigma = np.maximum((f['P90'].to_numpy() - f['P10'].to_numpy()) / (2 * 1.2816), EPS)
    sims = rng.normal(mean, sigma, size=(int(iterations), len(mean)))
    totals = sims.sum(axis=1)
    return {
        'paths': sims,
        'total_p10': float(np.quantile(totals, 0.10)),
        'total_p50': float(np.quantile(totals, 0.50)),
        'total_p90': float(np.quantile(totals, 0.90)),
        'prob_below_current_runrate': float(np.mean(totals < result['summary']['latest_month_revenue'] * len(mean)) * 100),
    }


def planning_text(result):
    s = result['summary']
    last = s['latest_month_revenue']
    avg_future = s['forecast_avg_month']
    delta = None if abs(last) <= EPS else (avg_future / last - 1) * 100
    if delta is None:
        direction = 'The latest actual month is zero, so percentage comparison is not meaningful.'
    elif delta > 5:
        direction = f'Average forecast monthly revenue is {delta:.1f}% above the latest actual month.'
    elif delta < -5:
        direction = f'Average forecast monthly revenue is {abs(delta):.1f}% below the latest actual month.'
    else:
        direction = f'Average forecast monthly revenue is broadly stable versus the latest actual month ({delta:+.1f}%).'
    return direction + f' Forecast risk is classified as {s["forecast_risk"].lower()} from rolling backtest error; use the P10–P90 range for operating plans rather than only the point forecast.'


def make_chart(result):
    monthly = result['monthly']
    forecast = result['forecast']
    future = pd.to_datetime(forecast['Month'])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(monthly.index.to_timestamp(), monthly, 'o-', lw=2, label='Actual revenue')
    if not result['holdout'].empty:
        held = pd.to_datetime(result['holdout']['Month'])
        ax.plot(held, result['holdout']['Hold-out prediction'], 's--', label='Recent backtest')
    ax.plot([monthly.index[-1].to_timestamp(), *future], [monthly.iloc[-1], *forecast['Forecast Net Revenue']], 'o--', lw=2, label='Ensemble forecast')
    ax.fill_between(future, forecast['P10'], forecast['P90'], alpha=.18, label='P10–P90 planning range')
    ax.fill_between(future, forecast['P2.5'], forecast['P97.5'], alpha=.08, label='P2.5–P97.5 outer range')
    ax.set(xlabel='Month', ylabel='Net Revenue (workbook units)')
    ax.xaxis.set_major_locator(MonthLocator(interval=max(1, len(monthly) // 10)))
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='best', frameon=False)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def report_markdown(result):
    s = result['summary']
    mape = 'Unavailable' if s['holdout_mape_pct'] is None else f"{s['holdout_mape_pct']:.2f}%"
    rows = '\n'.join(
        f"| {r['Month']} | {r['Forecast Net Revenue']:,.2f} | {r['P10']:,.2f} | {r['P90']:,.2f} |"
        for _, r in result['forecast'].iterrows()
    )
    ensemble = ', '.join(
        f"{r['Model']} ({r['Weight'] * 100:.0f}%)" for _, r in result['ensemble_members'].iterrows()
    )
    return f"""# Advanced revenue planning report

History: {s['historical_start']} to {s['historical_end']} ({s['history_months']} months).
Forecast method: **{s['forecast_method']}**. Best rolling-CV model: **{s['model']}**. Ensemble: {ensemble}.

| Month | Forecast | P10 | P90 |
|---|---:|---:|---:|
{rows}

Recent validation: MAPE {mape}; wMAPE {s['holdout_wmape_pct']:.2f}%; MAE {s['holdout_mae']:,.2f}; RMSE {s['holdout_rmse']:,.2f}; bias {s['holdout_bias']:,.2f}.
Forecast risk: **{s['forecast_risk']}**. The ranges are empirical planning intervals derived from rolling backtest error and recent volatility, not guaranteed confidence bounds.

## Planning implication

{planning_text(result)}
"""


def _to_excel(result, extra_tables=None):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result['history'].to_excel(writer, sheet_name='Historical', index=False)
        result['forecast'].to_excel(writer, sheet_name='Forecast', index=False)
        result['model_comparison'].to_excel(writer, sheet_name='Model CV', index=False)
        result['ensemble_members'].to_excel(writer, sheet_name='Ensemble', index=False)
        result['holdout'].to_excel(writer, sheet_name='Validation', index=False)
        result['anomalies'].to_excel(writer, sheet_name='Anomalies', index=False)
        pd.DataFrame([result['summary']]).to_excel(writer, sheet_name='Summary', index=False)
        if extra_tables:
            for name, df in extra_tables.items():
                safe = ''.join(ch for ch in name[:31] if ch not in '[]:*?/\\') or 'Table'
                df.to_excel(writer, sheet_name=safe, index=False)
    return output.getvalue()


def export_files(result, figure, extra_tables=None):
    png = BytesIO(); figure.savefig(png, format='png', dpi=180)
    files = {
        'forecast.csv': result['forecast'].to_csv(index=False).encode(),
        'monthly_actuals.csv': result['history'].to_csv(index=False).encode(),
        'recent_validation.csv': result['holdout'].to_csv(index=False).encode(),
        'rolling_cv_model_comparison.csv': result['model_comparison'].to_csv(index=False).encode(),
        'ensemble_weights.csv': result['ensemble_members'].to_csv(index=False).encode(),
        'anomalies.csv': result['anomalies'].to_csv(index=False).encode(),
        'analysis_summary.json': json.dumps(result['summary'], indent=2, allow_nan=False).encode(),
        'revenue_forecast.png': png.getvalue(),
        'report.md': report_markdown(result).encode(),
        'advanced_analysis.xlsx': _to_excel(result, extra_tables),
    }
    if extra_tables:
        for name, df in extra_tables.items():
            files[f'{name}.csv'] = df.to_csv(index=False).encode()
    archive = BytesIO()
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    files['forecast_outputs.zip'] = archive.getvalue()
    return files
