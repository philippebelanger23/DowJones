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

def calculate_cap_weighted_index(price_data, shares_data):
    # Identify standard rebalance dates (first trading day of each quarter)
    standard_rebalances = price_data.index.to_series().resample('BQS').first().dropna()
    
    # Specific dates of index composition changes
    event_dates = pd.to_datetime(['2020-08-31', '2024-02-26', '2024-11-08'])
    # Find the closest trading day for each event date
    event_trading_days = [price_data.index[price_data.index.searchsorted(d)] for d in event_dates if d in price_data.index or d < price_data.index.max()]
    
    # Combine and sort all rebalance dates
    rebalance_dates = pd.DatetimeIndex(sorted(list(set(standard_rebalances) | set(event_trading_days))))
    
    index_values = pd.Series(index=price_data.index, dtype=float)
    index_values.iloc[0] = 100.0
    
    current_units = None
    active_tickers = []
    
    for i in range(len(price_data)):
        date = price_data.index[i]
        
        # Check if it's a rebalance date (either standard or event)
        if date in rebalance_dates:
            # Update constituents for this period
            potential_tickers = get_constituents(date)
            active_tickers = [t for t in potential_tickers if t in price_data.columns]
            
            if len(active_tickers) < 30:
                print(f"Warning: Only {len(active_tickers)} tickers found for {date}")
                missing = [t for t in potential_tickers if t not in price_data.columns]
                print(f"Missing: {missing}")

            prices = price_data[active_tickers].iloc[i]
            shares = shares_data[active_tickers]
            
            # Rebalance logic
            market_caps = prices * shares
            weights = market_caps / market_caps.sum()
            
            total_value = index_values.iloc[i-1] if i > 0 else 100.0
            current_units = (weights * total_value) / prices
            
        # Daily calculation
        prices = price_data[active_tickers].iloc[i]
        index_values.iloc[i] = (current_units * prices).sum()
        
    return index_values

def main():
    data, shares = get_data(ALL_REQUIRED_TICKERS, PROXY_TICKER)
    cap_weighted = calculate_cap_weighted_index(data, shares)
    
    actual_dow = data[PROXY_TICKER].dropna()
    actual_dow = (actual_dow / actual_dow.iloc[0]) * 100
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(actual_dow.index, actual_dow, label='Actual Dow (Price-Weighted Proxy: DIA)', color='#003366', linewidth=2)
    plt.plot(cap_weighted.index, cap_weighted, label='Historical Cap-Weighted Dow (Dynamic Components)', color='#CC0000', linewidth=2, linestyle='--')
    
    plt.title('Dow Jones: Price-Weighted vs. Dynamic Cap-Weighted (Since 2020)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Normalized Index Value (Start = 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    total_ret_actual = (actual_dow.iloc[-1] / actual_dow.iloc[0] - 1) * 100
    total_ret_cap = (cap_weighted.iloc[-1] / cap_weighted.iloc[0] - 1) * 100
    
    stats_text = (f"Actual Dow Total Return: {total_ret_actual:.2f}%\n"
                  f"Cap-Weighted Dow Total Return: {total_ret_cap:.2f}%")
    plt.annotate(stats_text, xy=(0.05, 0.85), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('dow_comparison_chart.png')
    
    print(f"\nTotal Return (Since 2020):")
    print(f"Actual Dow (DIA): {total_ret_actual:.2f}%")
    print(f"Cap-Weighted Dow: {total_ret_cap:.2f}%")
    
    # Yearly Performance
    print("\nYearly Performance Comparison (Dynamic Components):")
    years = range(2020, 2027)
    yearly_data = []
    for year in years:
        year_start, year_end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
        if year == 2026: year_end = actual_dow.index.max()
        
        m_a, m_c = (actual_dow.index >= year_start) & (actual_dow.index <= year_end), (cap_weighted.index >= year_start) & (cap_weighted.index <= year_end)
        if m_a.any() and m_c.any():
            r_a = (actual_dow[m_a].iloc[-1] / actual_dow[m_a].iloc[0] - 1) * 100
            r_c = (cap_weighted[m_c].iloc[-1] / cap_weighted[m_c].iloc[0] - 1) * 100
            yearly_data.append({"Year": str(year) if year < 2026 else "2026 (YTD)", "Actual (%)": r_a, "Cap-Weighted (%)": r_c, "Diff (%)": r_c - r_a})
            
    print(pd.DataFrame(yearly_data).to_string(index=False, float_format="%.2f"))

if __name__ == "__main__":
    main()
