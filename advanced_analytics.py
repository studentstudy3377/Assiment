"""Reusable advanced analytics for driver, segment, and risk analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def build_driver_analysis(frame, revenue_col, date_col=None, max_features=30):
    work = frame.copy()
    y = pd.to_numeric(work[revenue_col], errors='coerce')
    candidates = []
    for c in work.columns:
        if c in {revenue_col, date_col}:
            continue
        num = pd.to_numeric(work[c], errors='coerce')
        if num.notna().mean() >= 0.75 and num.nunique(dropna=True) > 1:
            candidates.append(c)
    if not candidates or y.notna().sum() < 20:
        return pd.DataFrame(), None
    X = pd.DataFrame({c: pd.to_numeric(work[c], errors='coerce') for c in candidates})
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    # Keep continuous business measures even when most values are unique; only remove constants.
    X = X[[c for c in X.columns if X[c].nunique(dropna=True) > 1]]
    if X.empty or len(X) < 20:
        return pd.DataFrame(), None
    if X.shape[1] > max_features:
        corrs = X.corrwith(y).abs().sort_values(ascending=False).head(max_features)
        X = X[corrs.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    score = float(model.score(X_test, y_test)) if len(X_test) >= 2 else None
    perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, scoring='neg_mean_absolute_error')
    corr = X.corrwith(y)
    out = pd.DataFrame({
        'Driver': X.columns,
        'Permutation importance': perm.importances_mean,
        'Importance std': perm.importances_std,
        'Correlation with revenue': [corr.get(c, np.nan) for c in X.columns],
    }).sort_values('Permutation importance', ascending=False).reset_index(drop=True)
    return out, score


def segment_table(frame, category_col, revenue_col, top_n=20):
    temp = frame[[category_col, revenue_col]].copy()
    temp['_Revenue'] = pd.to_numeric(temp[revenue_col], errors='coerce')
    temp['_Category'] = temp[category_col].fillna('(Missing)').astype(str)
    out = temp.groupby('_Category')['_Revenue'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False).head(top_n)
    total = max(float(temp['_Revenue'].sum()), 1e-9)
    out['Revenue share (%)'] = out['sum'] / total * 100
    out['Cumulative share (%)'] = out['Revenue share (%)'].cumsum()
    out = out.reset_index().rename(columns={'_Category': category_col, 'sum': 'Revenue', 'mean': 'Avg revenue / row', 'count': 'Rows'})
    return out


def concentration_metrics(segment_df):
    if segment_df.empty or 'Revenue share (%)' not in segment_df:
        return {'top1_share': None, 'top5_share': None, 'hhi': None}
    shares = segment_df['Revenue share (%)'].to_numpy(dtype=float) / 100
    return {
        'top1_share': float(shares[0] * 100) if len(shares) else None,
        'top5_share': float(shares[:5].sum() * 100),
        'hhi': float(np.sum(shares ** 2) * 10000),
    }


def numeric_profile(frame, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(frame[c], errors='coerce')
        rows.append({
            'Column': c, 'Non-null': int(s.notna().sum()), 'Missing': int(s.isna().sum()),
            'Mean': float(s.mean()) if s.notna().any() else np.nan,
            'Median': float(s.median()) if s.notna().any() else np.nan,
            'Std dev': float(s.std()) if s.notna().sum() > 1 else np.nan,
            'Min': float(s.min()) if s.notna().any() else np.nan,
            'P25': float(s.quantile(.25)) if s.notna().any() else np.nan,
            'P75': float(s.quantile(.75)) if s.notna().any() else np.nan,
            'Max': float(s.max()) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)
