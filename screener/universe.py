"""Investable universe for the screener.

A fixed list of liquid U.S. large caps spread across all 11 GICS sectors, plus
the ETFs used for benchmarks and for the low-risk (operating reserve) sleeve.

Caveat: this is *today's* list of large caps, so a backtest on it carries
survivorship bias (companies that shrank or failed are missing). The backtest
therefore compares the screen against an equal-weight portfolio of the same
universe, which carries the identical bias, so the bias cancels out of the
selection-skill test.
"""

SECTORS: dict[str, list[str]] = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN", "IBM",
        "INTU", "TXN", "QCOM", "AMD", "AMAT", "MU", "ADI", "LRCX", "KLAC", "SNPS",
    ],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ORLY", "AZO", "CMG",
    ],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL", "MO", "PM", "KMB", "GIS"],
    "Health Care": [
        "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN",
        "ISRG", "SYK", "VRTX", "REGN", "BMY", "GILD", "MDT", "CVS",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
        "CB", "PGR", "AON", "V", "MA", "CME", "ICE",
    ],
    "Industrials": [
        "CAT", "DE", "HON", "UNP", "UPS", "GE", "RTX", "LMT", "BA", "ETN",
        "ITW", "WM", "GD", "NOC", "CSX", "EMR",
    ],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "OXY"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL"],
    "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG"],
    "Materials": ["LIN", "SHW", "APD", "ECL", "NEM", "FCX", "NUE"],
}

STOCKS: list[str] = [t for names in SECTORS.values() for t in names]
SECTOR_OF: dict[str, str] = {t: s for s, names in SECTORS.items() for t in names}

BENCHMARK = "SPY"
RISK_FREE = "^IRX"  # 13-week T-bill yield, in percent

# ETFs for Laura's strategic asset classes (used by the planning model and IPS).
ASSET_CLASS_ETFS: dict[str, str] = {
    "US equity": "VTI",
    "International equity": "VXUS",
    "Core bonds": "BND",
    "TIPS": "SCHP",
    "Short Treasuries": "SHY",
    "T-bills": "BIL",
}
