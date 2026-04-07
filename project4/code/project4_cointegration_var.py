"""
===============================================================================
PROJECT 4: Carbon Prices & Energy Stock Cointegration (VAR Analysis)
===============================================================================
RESEARCH QUESTION:
    Are carbon prices (KRBN), clean energy (ICLN), and fossil fuel (XLE) 
    stocks cointegrated? How do shocks transmit between markets?
METHOD:
    Johansen cointegration test, Vector Autoregression (VAR), 
    Granger causality, Impulse Response Functions
DATA:
    Yahoo Finance — KRBN (carbon ETF), ICLN, XLE
===============================================================================
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# STEP 1: Download data
# =============================================================================
print("STEP 1: Downloading data...")
tickers = {'KRBN':'Carbon Allowances','ICLN':'Clean Energy','XLE':'Fossil Fuels'}
prices = {}
for t in tickers:
    df = yf.download(t, start='2020-01-01', end='2025-12-31', auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    prices[t] = df['Close']
    print(f"  {t}: {len(df)} obs")

prices = pd.DataFrame(prices).dropna()
prices.to_csv('data/prices.csv')
returns = np.log(prices/prices.shift(1)).dropna() * 100
returns.to_csv('data/returns.csv')
print(f"  Combined: {len(prices)} trading days")

# =============================================================================
# STEP 2: Unit root tests
# =============================================================================
print("\nSTEP 2: ADF unit root tests...")
adf_rows = []
for col in prices.columns:
    # Levels
    r = adfuller(prices[col], autolag='AIC')
    adf_rows.append({'Series':col,'Transform':'Level','ADF_stat':r[0],'p_value':r[1],
                      'Stationary':'Yes' if r[1]<0.05 else 'No'})
    # First difference (returns)
    r2 = adfuller(returns[col], autolag='AIC')
    adf_rows.append({'Series':col,'Transform':'Returns','ADF_stat':r2[0],'p_value':r2[1],
                      'Stationary':'Yes' if r2[1]<0.05 else 'No'})
    print(f"  {col}: Level p={r[1]:.4f} ({'I(1)' if r[1]>0.05 else 'I(0)'}), "
          f"Returns p={r2[1]:.4f} ({'Stationary' if r2[1]<0.05 else 'Non-stat'})")

pd.DataFrame(adf_rows).to_csv('output/tables/adf_tests.csv', index=False)

# =============================================================================
# STEP 3: Johansen cointegration test
# =============================================================================
print("\nSTEP 3: Johansen cointegration test...")
joh = coint_johansen(prices, det_order=0, k_ar_diff=2)
print(f"  Trace statistics: {joh.lr1.round(2)}")
print(f"  Critical values (5%): {joh.cvt[:,1].round(2)}")
coint_results = pd.DataFrame({
    'Hypothesis': ['r=0','r≤1','r≤2'],
    'Trace_stat': joh.lr1.round(4),
    'Critical_5pct': joh.cvt[:,1].round(4),
    'Cointegrated': joh.lr1 > joh.cvt[:,1]
})
coint_results.to_csv('output/tables/cointegration_test.csv', index=False)

# =============================================================================
# STEP 4: VAR model
# =============================================================================
print("\nSTEP 4: Fitting VAR model...")
model = VAR(returns)
# Select optimal lag using AIC
lag_order = model.select_order(maxlags=15)
print(f"  Optimal lag by AIC: {lag_order.aic}")
optimal_lag = max(lag_order.aic, 2)
var_result = model.fit(optimal_lag)
print(f"  VAR({optimal_lag}) fitted. AIC={var_result.aic:.2f}")

# =============================================================================
# STEP 5: Granger Causality
# =============================================================================
print("\nSTEP 5: Granger causality tests...")
gc_rows = []
for col1 in returns.columns:
    for col2 in returns.columns:
        if col1 != col2:
            test = grangercausalitytests(returns[[col1, col2]], maxlag=5, verbose=False)
            min_p = min([test[lag][0]['ssr_ftest'][1] for lag in test])
            gc_rows.append({'Cause':col2,'Effect':col1,'Min_p':round(min_p,4),
                           'Significant':'Yes' if min_p<0.05 else 'No'})
            print(f"  {col2} → {col1}: p={min_p:.4f} {'***' if min_p<0.05 else ''}")

pd.DataFrame(gc_rows).to_csv('output/tables/granger_causality.csv', index=False)

# =============================================================================
# STEP 6: Impulse Response Functions
# =============================================================================
print("\nSTEP 6: Computing impulse response functions...")
irf = var_result.irf(periods=30)

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
names = list(returns.columns)
for i, target in enumerate(names):
    for j, shock in enumerate(names):
        ax = axes[i][j]
        irfs = irf.irfs[:, i, j]
        ax.plot(irfs, color='steelblue', linewidth=2)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.fill_between(range(len(irfs)), irfs, alpha=0.2, color='steelblue')
        ax.set_title(f'{shock} → {target}', fontsize=10, fontweight='bold')
        if i == 2: ax.set_xlabel('Days')
        if j == 0: ax.set_ylabel('Response')

plt.suptitle('Impulse Response Functions (30-day horizon)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig1_impulse_responses.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Price levels
fig, ax = plt.subplots(figsize=(14, 6))
norm = (prices / prices.iloc[0]) * 100
for col in norm.columns:
    ax.plot(norm.index, norm[col], label=tickers[col], linewidth=1.2)
ax.set_title('Normalized Prices (Base=100)', fontweight='bold')
ax.set_ylabel('Normalized Price')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('output/figures/fig2_prices.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Rolling correlations
fig, ax = plt.subplots(figsize=(14, 5))
rc1 = returns['KRBN'].rolling(60).corr(returns['ICLN'])
rc2 = returns['KRBN'].rolling(60).corr(returns['XLE'])
ax.plot(rc1.index, rc1, label='KRBN-ICLN', color='green')
ax.plot(rc2.index, rc2, label='KRBN-XLE', color='red')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('Rolling 60-day Correlations with Carbon Prices', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('output/figures/fig3_rolling_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n  COMPLETE!")
