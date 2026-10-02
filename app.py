"""Run with: python -m streamlit run app.py"""
from io import BytesIO
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from forecast_engine import analyze, make_chart, planning_text, export_files, simulate_forecast
from advanced_analytics import build_driver_analysis, segment_table, concentration_metrics, numeric_profile

st.set_page_config(page_title='Advanced Revenue Intelligence', page_icon='📊', layout='wide')

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(127,127,127,.06); border: 1px solid rgba(127,127,127,.15); padding: .8rem; border-radius: .7rem;}
.small-note {font-size:.86rem; opacity:.78;}
</style>
""", unsafe_allow_html=True)

st.title('Advanced Revenue Intelligence & Planning')
st.caption('Forecasting, rolling backtests, uncertainty, Monte Carlo scenarios, inventory policy, marketing economics, driver analysis, segmentation, anomalies, and all-column exploration.')


def parse_dates(series):
    raw = series.copy()
    numeric = raw.map(lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)))
    out = pd.to_datetime(raw.where(~numeric), format='mixed', errors='coerce')
    if numeric.any():
        out.loc[numeric] = pd.to_datetime(pd.to_numeric(raw.loc[numeric], errors='coerce'), unit='D', origin='1899-12-30', errors='coerce')
    return out


def fmt_num(v, decimals=1):
    if v is None or not np.isfinite(v):
        return 'N/A'
    if abs(v) >= 1e9: return f'{v/1e9:,.{decimals}f}B'
    if abs(v) >= 1e6: return f'{v/1e6:,.{decimals}f}M'
    if abs(v) >= 1e3: return f'{v/1e3:,.{decimals}f}K'
    return f'{v:,.{decimals}f}'


def guess_col(columns, keys):
    for c in columns:
        n = str(c).strip().lower().replace(' ', '_')
        if any(k in n for k in keys):
            return c
    return None


def safe_pct_change(a, b):
    if b is None or not np.isfinite(b) or abs(b) < 1e-9:
        return None
    return (a / b - 1) * 100


# ---------- Data selection ----------
with st.sidebar:
    st.header('Data setup')
    upload = st.file_uploader('Upload Excel workbook', type=['xlsx'])
    if upload is None:
        st.caption('Workbook should contain a date/month field and a revenue field. Other columns are analyzed automatically.')

if upload is None:
    st.info('Upload an Excel workbook from the sidebar to start.')
    st.stop()

contents = upload.getvalue(); source_name = upload.name
with st.sidebar:
    try:
        with pd.ExcelFile(BytesIO(contents)) as wb:
            sheet = st.selectbox('Sheet', wb.sheet_names, index=wb.sheet_names.index('Sheet1') if 'Sheet1' in wb.sheet_names else 0)
            data = pd.read_excel(wb, sheet_name=sheet)
    except Exception as exc:
        st.error(f'Workbook could not be read: {exc}')
        st.stop()

    cols = list(data.columns)
    if len(cols) < 2:
        st.error('The sheet needs at least two columns.')
        st.stop()
    normalized = {str(c).strip().upper().replace(' ', '_'): c for c in cols}
    date_default = normalized.get('MONTH', normalized.get('DATE', cols[0]))
    rev_default = normalized.get('NET_REVENUE', normalized.get('REVENUE', cols[-1]))
    date_col = st.selectbox('Date / month field', cols, index=cols.index(date_default))
    revenue_col = st.selectbox('Revenue field', cols, index=cols.index(rev_default))
    remove_duplicates = st.checkbox('Remove exact duplicate rows', False)

parsed = parse_dates(data[date_col]); valid = parsed.dropna()
if valid.empty:
    st.error('The selected date field contains no readable dates.')
    st.stop()
periods = parsed.dt.to_period('M')
months = pd.period_range(valid.min().to_period('M'), valid.max().to_period('M'), freq='M').astype(str).tolist()

with st.sidebar:
    st.header('Planning window')
    start_month = st.selectbox('From month', months, 0)
    end_options = [m for m in months if m >= start_month]
    end_month = st.selectbox('To month', end_options, len(end_options) - 1)
    horizon = st.slider('Forecast horizon (months)', 1, 18, 3)
    monte_carlo_runs = st.select_slider('Monte Carlo simulations', options=[1000, 2500, 5000, 10000], value=5000)

mask = (periods >= pd.Period(start_month, 'M')) & (periods <= pd.Period(end_month, 'M'))
selected = data.loc[mask].copy()
selected_periods = periods.loc[selected.index]
if selected.empty:
    st.error('No rows exist inside the selected period.')
    st.stop()

try:
    result = analyze(selected, date_col, revenue_col, remove_duplicates, horizon)
except Exception as exc:
    st.error(str(exc))
    st.dataframe(selected.head(30), hide_index=True, width='stretch')
    st.stop()

summary = result['summary']
summary.update(source_file=source_name, sheet=str(sheet), selected_start=start_month, selected_end=end_month)
monthly = result['monthly']; forecast = result['forecast']
mc = simulate_forecast(result, iterations=monte_carlo_runs)

# reusable column typing
numeric_view = selected.apply(pd.to_numeric, errors='coerce')
numeric_cols = [c for c in cols if c != date_col and numeric_view[c].notna().sum() >= max(1, int(selected[c].notna().sum() * 0.8))]
cat_cols = [c for c in cols if c not in numeric_cols and c != date_col]

st.caption(f'{source_name} · {sheet} · {start_month} → {end_month} · {summary["rows_used"]:,} rows · {summary["history_months"]} monthly observations')

# ---------- Executive KPIs ----------
latest = summary['latest_month_revenue']; total_fc = summary['forecast_total']
growth = summary['latest_mom_growth_pct']; yoy = summary['latest_yoy_growth_pct']; momentum = summary['three_month_momentum_pct']
row1 = st.columns(6)
row1[0].metric('Latest revenue', fmt_num(latest), None if growth is None else f'{growth:+.1f}% MoM')
row1[1].metric(f'{horizon}M forecast', fmt_num(total_fc), f'{safe_pct_change(summary["forecast_avg_month"], latest):+.1f}% avg vs latest' if safe_pct_change(summary['forecast_avg_month'], latest) is not None else None)
row1[2].metric('3M momentum', 'N/A' if momentum is None else f'{momentum:+.1f}%')
row1[3].metric('YoY growth', 'N/A' if yoy is None else f'{yoy:+.1f}%')
row1[4].metric('CV wMAPE', f'{summary["cv_best_wmape_pct"]:.1f}%')
row1[5].metric('Forecast risk', summary['forecast_risk'])

row2 = st.columns(5)
row2[0].metric('P10 total', fmt_num(mc['total_p10']))
row2[1].metric('P50 total', fmt_num(mc['total_p50']))
row2[2].metric('P90 total', fmt_num(mc['total_p90']))
row2[3].metric('Anomalies', summary['anomaly_count'])
completeness = (1 - selected.isna().sum().sum() / max(1, selected.size)) * 100
row2[4].metric('Data completeness', f'{completeness:.1f}%')

exec_tab, forecast_tab, risk_tab, inventory_tab, ads_tab, drivers_tab, quality_tab, allcols_tab = st.tabs([
    'Executive', 'Forecast lab', 'Risk & scenarios', 'Inventory', 'Marketing', 'Drivers & segments', 'Quality & anomalies', 'All columns'
])

# ---------- Executive ----------
with exec_tab:
    st.subheader('Executive planning view')
    c1, c2 = st.columns([1.55, 1])
    with c1:
        fig = make_chart(result)
        st.pyplot(fig, width='stretch')
    with c2:
        st.markdown('#### Planning interpretation')
        st.write(planning_text(result))
        st.markdown('#### Model blend')
        ensemble_show = result['ensemble_members'].copy()
        ensemble_show['Weight'] *= 100
        st.dataframe(ensemble_show.style.format({'Weight': '{:.1f}%'}), hide_index=True, width='stretch')
        st.markdown('#### Decision flags')
        flags = []
        if summary['forecast_risk'] == 'High': flags.append('Forecast uncertainty is high; use conservative inventory commitments and staged ad spend.')
        if momentum is not None and momentum < -10: flags.append('Recent 3-month momentum is materially negative versus the prior 3 months.')
        if growth is not None and abs(growth) > 25: flags.append('Latest month changed sharply versus the prior month; verify whether this is event-driven or a data issue.')
        if summary['anomaly_count'] > 0: flags.append(f'{summary["anomaly_count"]} historical month(s) were flagged as possible anomalies.')
        if completeness < 95: flags.append('Workbook completeness is below 95%; missing fields may weaken driver analysis.')
        if not flags: flags = ['No major automated warning flags were triggered for the selected period.']
        for f in flags: st.write('• ' + f)

    st.markdown('#### Forecast operating table')
    fc_view = forecast.copy()
    fc_view['Upside vs P10'] = fc_view['Forecast Net Revenue'] - fc_view['P10']
    fc_view['P90 upside'] = fc_view['P90'] - fc_view['Forecast Net Revenue']
    st.dataframe(fc_view.style.format({c: '{:,.2f}' for c in fc_view.columns if c != 'Month'}), hide_index=True, width='stretch')

# ---------- Forecast lab ----------
with forecast_tab:
    st.subheader('Forecast model laboratory')
    st.write('Models are evaluated with rolling-origin one-step backtests. The final forecast blends up to the three best models using inverse-error weights, reducing dependence on a single model.')
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('#### Rolling-CV model comparison')
        st.dataframe(result['model_comparison'].style.format({
            'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'MAPE (%)': '{:.2f}', 'sMAPE (%)': '{:.2f}',
            'wMAPE (%)': '{:.2f}', 'Bias': '{:,.2f}'
        }, na_rep='—'), hide_index=True, width='stretch')
    with right:
        st.markdown('#### Recent backtest')
        st.dataframe(result['holdout'].style.format({c: '{:,.2f}' for c in result['holdout'].columns if c != 'Month'}), hide_index=True, width='stretch')
        m1, m2, m3 = st.columns(3)
        m1.metric('Recent MAE', fmt_num(summary['holdout_mae']))
        m2.metric('Recent wMAPE', f'{summary["holdout_wmape_pct"]:.1f}%')
        m3.metric('Bias', fmt_num(summary['holdout_bias']))

    st.markdown('#### Forecast uncertainty')
    st.dataframe(forecast.style.format({c: '{:,.2f}' for c in forecast.columns if c != 'Month'}), hide_index=True, width='stretch')
    st.caption('P10–P90 and P2.5–P97.5 are empirical planning ranges calibrated from rolling backtest error and recent volatility. They are decision ranges, not guarantees.')

# ---------- Risk & scenarios ----------
with risk_tab:
    st.subheader('Scenario and risk simulator')
    a, b, c, d = st.columns(4)
    downside = a.slider('Downside shock', -60, 0, -15, 1, format='%d%%')
    upside = b.slider('Upside shock', 0, 60, 15, 1, format='%d%%')
    price_change = c.slider('Price / mix change', -30, 30, 0, 1, format='%d%%')
    demand_change = d.slider('Demand change', -30, 30, 0, 1, format='%d%%')

    scenario = pd.DataFrame({'Month': forecast['Month']})
    base_factor = (1 + price_change / 100) * (1 + demand_change / 100)
    scenario['Downside'] = forecast['Forecast Net Revenue'] * base_factor * (1 + downside / 100)
    scenario['Base'] = forecast['Forecast Net Revenue'] * base_factor
    scenario['Upside'] = forecast['Forecast Net Revenue'] * base_factor * (1 + upside / 100)
    st.line_chart(scenario.set_index('Month'))
    totals = scenario[['Downside', 'Base', 'Upside']].sum()
    x1, x2, x3, x4 = st.columns(4)
    x1.metric('Downside total', fmt_num(totals['Downside']))
    x2.metric('Base total', fmt_num(totals['Base']))
    x3.metric('Upside total', fmt_num(totals['Upside']))
    x4.metric('Monte Carlo below current run-rate', f'{mc["prob_below_current_runrate"]:.1f}%')

    st.markdown('#### Monte Carlo distribution summary')
    mc_df = pd.DataFrame({
        'Measure': ['P10 total', 'P50 total', 'P90 total', 'Point forecast total'],
        'Value': [mc['total_p10'], mc['total_p50'], mc['total_p90'], summary['forecast_total']]
    })
    st.dataframe(mc_df.style.format({'Value': '{:,.2f}'}), hide_index=True, width='stretch')

# ---------- Inventory ----------
with inventory_tab:
    st.subheader('Inventory policy simulator')
    st.caption('Translate revenue into unit demand and calculate a service-level-based safety stock and reorder point.')
    a, b, c, d, e = st.columns(5)
    avg_price = a.number_input('Average selling price / unit', min_value=0.01, value=100.0, step=10.0)
    unit_cost = b.number_input('Unit cost', min_value=0.0, value=50.0, step=5.0)
    lead_days = c.number_input('Lead time (days)', min_value=1, value=30, step=1)
    service_level = d.selectbox('Target service level', ['90%', '95%', '97.5%', '99%'], index=1)
    current_stock = e.number_input('Current usable stock', min_value=0.0, value=0.0, step=100.0)
    z_map = {'90%': 1.282, '95%': 1.645, '97.5%': 1.960, '99%': 2.326}
    z = z_map[service_level]

    hist_units = monthly / avg_price
    monthly_unit_std = float(hist_units.diff().dropna().std()) if len(hist_units) > 2 else float(hist_units.std())
    daily_std = monthly_unit_std / math.sqrt(30.44) if np.isfinite(monthly_unit_std) else 0.0
    safety_stock = z * daily_std * math.sqrt(lead_days)

    inv_plan = forecast[['Month', 'Forecast Net Revenue', 'P10', 'P90']].copy()
    inv_plan['Expected units'] = inv_plan['Forecast Net Revenue'] / avg_price
    inv_plan['P90 demand units'] = inv_plan['P90'] / avg_price
    inv_plan['Lead-time demand'] = inv_plan['Expected units'] * (lead_days / 30.44)
    inv_plan['Safety stock'] = safety_stock
    inv_plan['Reorder point'] = inv_plan['Lead-time demand'] + safety_stock
    inv_plan['Suggested order qty'] = np.maximum(inv_plan['P90 demand units'] + safety_stock - current_stock, 0)
    inv_plan['Inventory purchase value'] = inv_plan['Suggested order qty'] * unit_cost
    inv_plan['Projected gross profit'] = inv_plan['Forecast Net Revenue'] - inv_plan['Expected units'] * unit_cost
    st.dataframe(inv_plan.style.format({c: '{:,.2f}' for c in inv_plan.columns if c != 'Month'}), hide_index=True, width='stretch')

    i1, i2, i3, i4 = st.columns(4)
    i1.metric('Safety stock', f'{safety_stock:,.0f} units')
    i2.metric('Total suggested order', f'{inv_plan["Suggested order qty"].sum():,.0f} units')
    i3.metric('Purchase value', fmt_num(inv_plan['Inventory purchase value'].sum()))
    i4.metric('Forecast gross profit', fmt_num(inv_plan['Projected gross profit'].sum()))

    inv_candidates = [c for c in numeric_cols if any(k in str(c).lower() for k in ['unit', 'qty', 'quantity', 'stock', 'inventory', 'order', 'sold', 'return'])]
    if inv_candidates:
        st.markdown('#### Detected inventory-related workbook fields')
        inv_hist = selected.assign(_Month=selected_periods.astype(str)).groupby('_Month')[inv_candidates].sum(numeric_only=True)
        st.dataframe(inv_hist.style.format('{:,.2f}'), width='stretch')

# ---------- Marketing ----------
with ads_tab:
    st.subheader('Marketing economics & budget planner')
    a, b, c, d, e = st.columns(5)
    gross_margin_pct = a.slider('Gross margin %', 1, 100, 40)
    target_roas = b.number_input('Target ROAS', min_value=0.1, value=4.0, step=0.1)
    conversion_rate = c.number_input('Conversion rate %', min_value=0.01, value=2.5, step=0.1)
    avg_order_value = d.number_input('Average order value', min_value=0.01, value=100.0, step=10.0)
    max_budget_pct = e.slider('Budget cap as % revenue', 1, 80, 20)

    ad_plan = forecast[['Month', 'Forecast Net Revenue', 'P10', 'P90']].copy()
    ad_plan['Gross margin'] = ad_plan['Forecast Net Revenue'] * gross_margin_pct / 100
    ad_plan['Budget from target ROAS'] = ad_plan['Forecast Net Revenue'] / target_roas
    ad_plan['Budget cap'] = ad_plan['Forecast Net Revenue'] * max_budget_pct / 100
    ad_plan['Recommended budget ceiling'] = ad_plan[['Budget from target ROAS', 'Budget cap']].min(axis=1)
    ad_plan['Break-even ROAS'] = 100 / gross_margin_pct
    ad_plan['Expected orders'] = ad_plan['Forecast Net Revenue'] / avg_order_value
    ad_plan['Expected conversions from clicks'] = ad_plan['Expected orders']
    ad_plan['Indicative clicks needed'] = ad_plan['Expected orders'] / (conversion_rate / 100)
    ad_plan['Max CPC at target economics'] = ad_plan['Recommended budget ceiling'] / ad_plan['Indicative clicks needed'].replace(0, np.nan)
    ad_plan['Budget / gross margin (%)'] = ad_plan['Recommended budget ceiling'] / ad_plan['Gross margin'].replace(0, np.nan) * 100
    st.dataframe(ad_plan.style.format({c: '{:,.2f}' for c in ad_plan.columns if c != 'Month'}), hide_index=True, width='stretch')

    a1, a2, a3, a4 = st.columns(4)
    a1.metric('Total budget ceiling', fmt_num(ad_plan['Recommended budget ceiling'].sum()))
    a2.metric('Break-even ROAS', f'{100/gross_margin_pct:.2f}x')
    a3.metric('Target ROAS buffer', f'{target_roas - 100/gross_margin_pct:+.2f}x')
    a4.metric('Expected orders', f'{ad_plan["Expected orders"].sum():,.0f}')

    ad_candidates = [c for c in numeric_cols if any(k in str(c).lower() for k in ['ad', 'spend', 'cost', 'click', 'impression', 'roas', 'cpc', 'cpm', 'conversion', 'campaign'])]
    if ad_candidates:
        st.markdown('#### Historical marketing signals detected in workbook')
        ad_hist = selected.assign(_Month=selected_periods.astype(str)).groupby('_Month')[ad_candidates].sum(numeric_only=True)
        st.dataframe(ad_hist.style.format('{:,.2f}'), width='stretch')
        spend_guess = guess_col(ad_candidates, ['spend', 'ad_cost', 'advertising_cost', 'media_cost'])
        if spend_guess is not None and ad_hist[spend_guess].sum() > 0:
            hist_rev = pd.to_numeric(selected[revenue_col], errors='coerce').groupby(selected_periods).sum()
            hist_rev.index = hist_rev.index.astype(str)
            aligned = pd.concat([hist_rev.rename('Revenue'), ad_hist[spend_guess].rename('Ad spend')], axis=1).dropna()
            observed_roas = aligned['Revenue'].sum() / aligned['Ad spend'].sum()
            st.metric('Observed aggregate ROAS', f'{observed_roas:.2f}x')
            st.caption('Observed ROAS is descriptive and does not prove advertising caused the revenue.')

# ---------- Drivers & segments ----------
with drivers_tab:
    st.subheader('Revenue drivers and concentration')
    with st.spinner('Calculating driver signals...'):
        driver_df, driver_score = build_driver_analysis(selected, revenue_col, date_col)
    if driver_df.empty:
        st.info('Driver modeling needs at least ~20 usable rows and one additional numeric predictor.')
    else:
        d1, d2 = st.columns([1.5, 1])
        with d1:
            st.markdown('#### Numeric driver ranking')
            st.dataframe(driver_df.head(15).style.format({
                'Permutation importance': '{:,.3f}', 'Importance std': '{:,.3f}', 'Correlation with revenue': '{:+.3f}'
            }), hide_index=True, width='stretch')
        with d2:
            st.metric('Driver model holdout R²', 'N/A' if driver_score is None else f'{driver_score:.2f}')
            st.write('Permutation importance estimates which numeric variables help predict row-level revenue in this dataset.')
            st.caption('Predictive importance and correlation are associations, not causal effects.')

    if cat_cols:
        st.markdown('#### Segment contribution & concentration')
        category = st.selectbox('Segment dimension', cat_cols, key='segment_dimension')
        seg = segment_table(selected, category, revenue_col, top_n=25)
        metrics = concentration_metrics(seg)
        s1, s2, s3 = st.columns(3)
        s1.metric('Top segment share', 'N/A' if metrics['top1_share'] is None else f'{metrics["top1_share"]:.1f}%')
        s2.metric('Top 5 share', 'N/A' if metrics['top5_share'] is None else f'{metrics["top5_share"]:.1f}%')
        s3.metric('HHI concentration', 'N/A' if metrics['hhi'] is None else f'{metrics["hhi"]:.0f}')
        st.dataframe(seg.style.format({'Revenue': '{:,.2f}', 'Avg revenue / row': '{:,.2f}', 'Revenue share (%)': '{:.1f}%', 'Cumulative share (%)': '{:.1f}%'}), hide_index=True, width='stretch')
        if not seg.empty:
            st.bar_chart(seg.set_index(category)['Revenue'])
    else:
        seg = pd.DataFrame()

# ---------- Quality ----------
with quality_tab:
    st.subheader('Data quality, anomalies & governance checks')
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric('Rows', f'{len(selected):,}')
    q2.metric('Exact duplicates', f'{selected.duplicated().sum():,}')
    q3.metric('Missing cells', f'{selected.isna().sum().sum():,}')
    q4.metric('Columns', len(selected.columns))
    q5.metric('Completeness', f'{completeness:.1f}%')

    quality = pd.DataFrame({
        'Column': selected.columns,
        'Missing': selected.isna().sum().values,
        'Missing %': selected.isna().mean().values * 100,
        'Unique': selected.nunique(dropna=True).values,
        'Constant?': [selected[c].nunique(dropna=True) <= 1 for c in selected.columns],
    })
    st.dataframe(quality.style.format({'Missing %': '{:.1f}%'}), hide_index=True, width='stretch')

    st.markdown('#### Monthly revenue anomalies')
    anomalies = result['anomalies']
    if anomalies.empty:
        st.success('No strong monthly anomalies were detected by the robust local-baseline rule.')
    else:
        st.dataframe(anomalies.style.format({'Revenue': '{:,.2f}', 'Expected': '{:,.2f}', 'Residual': '{:,.2f}', 'Robust z-score': '{:+.2f}'}), hide_index=True, width='stretch')
        st.caption('Anomaly flags identify unusual observations for review; they do not automatically mean the data is wrong.')

    st.markdown('#### Structural checks')
    warnings = []
    if selected.duplicated().sum() > 0: warnings.append('Exact duplicate rows exist in the selected range.')
    if any(quality['Constant?']): warnings.append('One or more columns are constant and provide no analytical variation.')
    high_missing = quality.loc[quality['Missing %'] >= 30, 'Column'].astype(str).tolist()
    if high_missing: warnings.append('Columns with ≥30% missing values: ' + ', '.join(high_missing[:10]))
    if len(monthly) < 12: warnings.append('Fewer than 12 monthly observations are available, so annual seasonality cannot be assessed reliably.')
    if not warnings: warnings.append('No major structural quality issues were detected by the automated checks.')
    for w in warnings: st.write('• ' + w)

# ---------- All columns ----------
with allcols_tab:
    st.subheader('All-column analytical explorer')
    profile_rows = []
    for col in cols:
        s = selected[col]
        num = pd.to_numeric(s, errors='coerce')
        is_num = col in numeric_cols
        profile_rows.append({
            'Column': str(col), 'Detected type': 'Numeric' if is_num else str(s.dtype),
            'Non-null': int(s.notna().sum()), 'Missing': int(s.isna().sum()), 'Unique': int(s.nunique(dropna=True)),
            'Sum': float(num.sum()) if is_num else np.nan, 'Average': float(num.mean()) if is_num else np.nan,
            'Median': float(num.median()) if is_num else np.nan, 'Min': float(num.min()) if is_num else np.nan,
            'Max': float(num.max()) if is_num else np.nan,
        })
    profile = pd.DataFrame(profile_rows)
    st.dataframe(profile.style.format({'Sum': '{:,.2f}', 'Average': '{:,.2f}', 'Median': '{:,.2f}', 'Min': '{:,.2f}', 'Max': '{:,.2f}'}, na_rep='—'), hide_index=True, width='stretch')

    if numeric_cols:
        st.markdown('#### Numeric deep dive')
        metric = st.selectbox('Numeric metric', numeric_cols)
        agg = st.radio('Monthly aggregation', ['Sum', 'Average', 'Median', 'Min', 'Max'], horizontal=True)
        temp = selected.assign(_Month=selected_periods.astype(str), _Metric=pd.to_numeric(selected[metric], errors='coerce'))
        agg_fn = {'Sum': 'sum', 'Average': 'mean', 'Median': 'median', 'Min': 'min', 'Max': 'max'}[agg]
        monthly_metric = temp.groupby('_Month')['_Metric'].agg(agg_fn)
        st.line_chart(monthly_metric)
        nprof = numeric_profile(selected, numeric_cols)
        st.dataframe(nprof.style.format({c: '{:,.2f}' for c in ['Mean', 'Median', 'Std dev', 'Min', 'P25', 'P75', 'Max']}), hide_index=True, width='stretch')

    if cat_cols:
        st.markdown('#### Categorical deep dive')
        cat = st.selectbox('Categorical field', cat_cols, key='allcat')
        counts = selected[cat].fillna('(Missing)').astype(str).value_counts().head(30).rename_axis(cat).reset_index(name='Rows')
        st.bar_chart(counts.set_index(cat))
        st.dataframe(counts, hide_index=True, width='stretch')

# ---------- Export ----------
extra = {
    'Scenario Analysis': scenario,
    'Inventory Plan': inv_plan,
    'Marketing Plan': ad_plan,
    'Quality Profile': quality,
    'Column Profile': profile,
}
if 'driver_df' in locals() and not driver_df.empty: extra['Driver Analysis'] = driver_df
if 'seg' in locals() and not seg.empty: extra['Segment Analysis'] = seg

files = export_files(result, fig, extra)
plt.close(fig)
st.divider()
st.subheader('Export complete decision pack')
c1, c2 = st.columns([1, 2])
with c1:
    st.download_button('Download advanced analysis ZIP', files['forecast_outputs.zip'], 'advanced_revenue_intelligence_outputs.zip', 'application/zip', type='primary')
with c2:
    st.caption('Includes Excel workbook, forecast CSV, rolling-CV results, ensemble weights, anomalies, scenario plan, inventory plan, marketing plan, chart, JSON summary, and Markdown report.')
