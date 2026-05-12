import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Historical Index Composition Changes:
# Aug 31, 2020: CRM, AMGN, HON replace XOM, PFE, RTX
# Feb 26, 2024: AMZN replaces WBA
# Nov 8, 2024:  NVDA replaces INTC, SHW replaces DOW

CURRENT_TICKERS = [
    'GS', 'CAT', 'MSFT', 'UNH', 'AMGN', 'V', 'SHW', 'AXP', 'HD', 'JPM',
    'TRV', 'AAPL', 'MCD', 'AMZN', 'BA', 'IBM', 'JNJ', 'NVDA', 'HON', 'CVX',
    'CRM', 'PG', 'MMM', 'WMT', 'MRK', 'DIS', 'CSCO', 'KO', 'VZ', 'NKE'
]
HISTORICAL_TICKERS = ['INTC', 'DOW', 'WBA', 'XOM', 'PFE', 'RTX']
ALL_REQUIRED_TICKERS = list(set(CURRENT_TICKERS + HISTORICAL_TICKERS))
PROXY_TICKER = 'DIA'

def get_constituents(date):
    """Returns the list of 30 DJIA tickers for a specific date."""
    # Start with a base (the list as of early 2020)
    # Base: Before Aug 31, 2020
    tickers = [
        'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS', 'HD', 
        'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 
        'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'WBA', 'XOM', 'PFE', 'RTX'
    ]
    
    dt = pd.to_datetime(date)
    
    # Apply changes chronologically
    if dt >= pd.to_datetime('2020-08-31'):
        # CRM, AMGN, HON replace XOM, PFE, RTX
        tickers = [t for t in tickers if t not in ['XOM', 'PFE', 'RTX']]
        tickers.extend(['CRM', 'AMGN', 'HON'])
        
    if dt >= pd.to_datetime('2024-02-26'):
        # AMZN replaces WBA
        tickers = [t for t in tickers if t != 'WBA']
        tickers.append('AMZN')
        
    if dt >= pd.to_datetime('2024-11-08'):
        # NVDA replaces INTC, SHW replaces DOW
        tickers = [t for t in tickers if t not in ['INTC', 'DOW']]
        tickers.extend(['NVDA', 'SHW'])
        
    return tickers

def get_data(tickers, proxy):
    start_date = "2020-01-01"
    end_date = datetime.now()
    
    print(f"Downloading data for {len(tickers)} potential tickers and {proxy}...")
    data_all = yf.download(tickers + [proxy], start=start_date, end=end_date, auto_adjust=False)
    
    if isinstance(data_all.columns, pd.MultiIndex):
        data = data_all['Adj Close']
    else:
        data = data_all
    
    shares = {}
    print("Fetching current shares outstanding for weighting...")
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            shares[t] = ticker_obj.info.get('sharesOutstanding', 0)
        except Exception:
            shares[t] = 0
            
    return data, pd.Series(shares)

def calculate_indices_and_concentration(price_data, shares_data):
    # Identify standard rebalance dates (first trading day of each quarter)
    standard_rebalances = price_data.index.to_series().resample('BQS').first().dropna()
    
    # Specific dates of index composition changes
    event_dates = pd.to_datetime(['2020-08-31', '2024-02-26', '2024-11-08'])
    event_trading_days = [price_data.index[price_data.index.searchsorted(d)] for d in event_dates if d in price_data.index or d < price_data.index.max()]
    
    rebalance_dates = pd.DatetimeIndex(sorted(list(set(standard_rebalances) | set(event_trading_days))))
    
    index_values = pd.Series(index=price_data.index, dtype=float)
    index_values.iloc[0] = 100.0
    
    # Data structures for concentration tracking
    conc_data = []
    
    current_units = None
    active_tickers = []
    
    for i in range(len(price_data)):
        date = price_data.index[i]
        prices = price_data.iloc[i]
        
        # Check if it's a rebalance date
        if date in rebalance_dates:
            active_tickers = get_constituents(date)
            active_tickers = [t for t in active_tickers if t in price_data.columns]
            
            p_active = prices[active_tickers]
            s_active = shares_data[active_tickers]
            
            # Cap-Weighted Weights
            mcaps = p_active * s_active
            w_cap = mcaps / mcaps.sum()
            
            # Price-Weighted Weights (The actual Dow)
            w_price = p_active / p_active.sum()
            
            # Track Concentration (Top 10 - More robust for 30-stock index)
            top10_cap = w_cap.sort_values(ascending=False).head(10).sum() * 100
            top10_price = w_price.sort_values(ascending=False).head(10).sum() * 100
            
            conc_data.append({
                'Date': date,
                'Cap-Weighted Top 10 (%)': top10_cap,
                'Price-Weighted Top 10 (%)': top10_price
            })
            
            total_value = index_values.iloc[i-1] if i > 0 else 100.0
            current_units = (w_cap * total_value) / p_active
            
        # Daily calculation
        p_active = prices[active_tickers]
        index_values.iloc[i] = (current_units * p_active).sum()
        
    return index_values, pd.DataFrame(conc_data)

def main():
    data, shares = get_data(ALL_REQUIRED_TICKERS, PROXY_TICKER)
    cap_weighted, concentration = calculate_indices_and_concentration(data, shares)
    
    actual_dow = data[PROXY_TICKER].dropna()
    actual_dow = (actual_dow / actual_dow.iloc[0]) * 100
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Chart 1: Performance
    ax1.plot(actual_dow.index, actual_dow, label='Actual Dow (Price-Weighted Proxy: DIA)', color='#003366', linewidth=2)
    ax1.plot(cap_weighted.index, cap_weighted, label='Historical Cap-Weighted Dow (Dynamic)', color='#CC0000', linewidth=2, linestyle='--')
    ax1.set_title('Dow Jones: Performance Comparison (Since 2020)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized Value (Start = 100)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    total_ret_actual = (actual_dow.iloc[-1] / actual_dow.iloc[0] - 1) * 100
    total_ret_cap = (cap_weighted.iloc[-1] / cap_weighted.iloc[0] - 1) * 100
    
    stats_text = (f"Actual Dow Total Return: {total_ret_actual:.2f}%\n"
                  f"Cap-Weighted Dow Total Return: {total_ret_cap:.2f}%")
    ax1.annotate(stats_text, xy=(0.05, 0.85), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Chart 2: Concentration (Top 10 Weight)
    ax2.plot(concentration['Date'], concentration['Price-Weighted Top 10 (%)'], label='Price-Weighted Top 10 Concentration (%)', color='#003366', alpha=0.6)
    ax2.plot(concentration['Date'], concentration['Cap-Weighted Top 10 (%)'], label='Cap-Weighted Top 10 Concentration (%)', color='#CC0000', alpha=0.8)
    ax2.set_title('Index Concentration: Combined Weight of Top 10 Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weight (%)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(50, 100) # Adjusted scale for Top 10 concentration
    
    # Add Data Labels over the curves (one per year)
    years_to_label = range(2020, 2027)
    for year in years_to_label:
        label_date = pd.Timestamp(year, 7, 1) # Mid-year label
        if year == 2026: label_date = actual_dow.index.max()
        
        if label_date in actual_dow.index or label_date < actual_dow.index.max():
            idx = actual_dow.index.searchsorted(label_date)
            if idx < len(actual_dow.index):
                trading_day = actual_dow.index[idx]
                
                # Chart 1: Performance
                val_actual = actual_dow.loc[trading_day]
                val_cap = cap_weighted.loc[trading_day]
                ax1.text(trading_day, val_actual + 5, f'{val_actual:.0f}', fontsize=9, color='#003366', fontweight='bold', ha='center', va='bottom')
                ax1.text(trading_day, val_cap + 5, f'{val_cap:.0f}', fontsize=9, color='#CC0000', fontweight='bold', ha='center', va='bottom')

                # Chart 2: Concentration
                if trading_day in concentration['Date'].values:
                    c_row = concentration[concentration['Date'] == trading_day].iloc[0]
                    c_actual = c_row['Price-Weighted Top 10 (%)']
                    c_cap = c_row['Cap-Weighted Top 10 (%)']
                    ax2.text(trading_day, c_actual + 2, f'{c_actual:.0f}%', fontsize=9, color='#003366', fontweight='bold', ha='center', va='bottom')
                    ax2.text(trading_day, c_cap + 2, f'{c_cap:.0f}%', fontsize=9, color='#CC0000', fontweight='bold', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('dow_comparison_chart.png')
    
    print(f"\nTotal Return (Since 2020):")
    print(f"Actual Dow (DIA): {total_ret_actual:.2f}%")
    print(f"Cap-Weighted Dow: {total_ret_cap:.2f}%")
    
    # Yearly Performance
    print("\nYearly Performance Comparison (Dynamic Components):")
    years = range(2020, 2027)
    yearly_data = []
    
    # Calculate daily returns for metrics
    actual_returns = actual_dow.pct_change().dropna()
    cap_returns = cap_weighted.pct_change().dropna()
    
    for year in years:
        year_start, year_end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
        if year == 2026: year_end = actual_dow.index.max()
        
        m_a = (actual_dow.index >= year_start) & (actual_dow.index <= year_end)
        m_c = (cap_weighted.index >= year_start) & (cap_weighted.index <= year_end)
        
        if m_a.any() and m_c.any():
            # Returns
            r_a = (actual_dow[m_a].iloc[-1] / actual_dow[m_a].iloc[0] - 1) * 100
            r_c = (cap_weighted[m_c].iloc[-1] / cap_weighted[m_c].iloc[0] - 1) * 100
            
            # Sharpe (Annualized, assuming 0% risk-free rate)
            ret_a_year = actual_returns[actual_returns.index.year == year]
            ret_c_year = cap_returns[cap_returns.index.year == year]
            
            sharpe_a = (ret_a_year.mean() / ret_a_year.std() * np.sqrt(252)) if len(ret_a_year) > 1 else 0
            sharpe_c = (ret_c_year.mean() / ret_c_year.std() * np.sqrt(252)) if len(ret_c_year) > 1 else 0
            
            # Max Drawdown
            dd_a = (actual_dow[m_a] / actual_dow[m_a].cummax() - 1).min() * 100
            dd_c = (cap_weighted[m_c] / cap_weighted[m_c].cummax() - 1).min() * 100
            
            yearly_data.append({
                "Year": str(year) if year < 2026 else "2026 (YTD)",
                "Actual Ret (%)": r_a,
                "CapW Ret (%)": r_c,
                "Actual Sharpe": sharpe_a,
                "CapW Sharpe": sharpe_c,
                "Actual MaxDD (%)": dd_a,
                "CapW MaxDD (%)": dd_c
            })
            
    print(pd.DataFrame(yearly_data).to_string(index=False, float_format="%.2f"))

if __name__ == "__main__":
    main()
