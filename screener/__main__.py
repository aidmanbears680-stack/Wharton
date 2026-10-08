"""Command line: python -m screener {report,screen,plan}."""

import argparse

from . import laura_plan as LP


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m screener", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="full run: method comparison, plan, trade notes -> reports/")
    r.add_argument("--refresh", action="store_true", help="re-download prices")
    r.add_argument("--capital", type=float, default=100_000.0, help="dollars for position sizing")
    s = sub.add_parser("screen", help="today's picks with per-trade statistics -> reports/TRADING_NOTES.md")
    s.add_argument("--refresh", action="store_true")
    s.add_argument("--capital", type=float, default=100_000.0)
    p = sub.add_parser("plan", help="print Laura plan summary and the 2031 co-sponsor range")
    p.add_argument("--sleeve-2031", type=float, help="observed growth-sleeve value in Jan 2031")
    args = ap.parse_args()

    if args.cmd == "report":
        from .report import run

        run(refresh=args.refresh, capital=args.capital)
    elif args.cmd == "screen":
        from .data import load_prices, risk_free_daily
        from .report import OUT
        from .trade_notes import build_notes
        from .universe import BENCHMARK, RISK_FREE, STOCKS

        prices = load_prices(STOCKS + [BENCHMARK, RISK_FREE, "BIL"], refresh=args.refresh)
        table, md = build_notes(prices, risk_free_daily(prices), STOCKS, args.capital)
        OUT.mkdir(exist_ok=True)
        (OUT / "TRADING_NOTES.md").write_text(md)
        print(table[["sector", "verdict", "weight", "dollars", "shares", "price", "stop"]].round(3).to_string())
    elif args.cmd == "plan":
        for name in LP.STRATEGIES:
            df = LP.simulate_plan(name)
            c = df["contribution"]
            print(f"{name:45s} P(funded) {df['reserve_funded'].mean():6.1%}  contribution "
                  f"p10 ${c.quantile(.1):>9,.0f}  median ${c.median():>9,.0f}  p90 ${c.quantile(.9):>9,.0f}")
        if args.sleeve_2031:
            lo, mid, hi = LP.cosponsor_range(LP.simulate_plan(), args.sleeve_2031)
            print(f"\nQuote to co-sponsors: ${lo:,.0f} to ${hi:,.0f} (median ${mid:,.0f}), 80% confidence")


if __name__ == "__main__":
    main()
