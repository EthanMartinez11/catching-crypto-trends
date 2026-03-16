# Catching Crypto Trends — Section 7

Paper-style implementation of the diversified program described in Section 7 of *Catching Crypto Trends*.

## Idea

The strategy applies a long-only trend-following program across a monthly-selected crypto universe.
Position sizing follows volatility targeting, with transaction costs, leverage caps, monthly universe updates and in-month liquidity exits.

## Main rules

- Universe: monthly top-volume crypto universe from a CSV file
- Trading month uses the previous month's selected universe
- Signal: Donchian breakout on close
- Lookbacks: 10, 20, 30, 60, 90, 150
- Volatility targeting with leverage cap
- Transaction costs: 10 bps on turnover
- Rebalance threshold: 20% for volatility-driven resizing only
- Monthly equal-weight sleeve allocation across selected assets
- In-month liquidity exits based on rolling volume and median absolute return

## Outputs

The script saves:
- results.txt
- equity plots vs BTC
- drawdown and rolling vol plots
- monthly and yearly return charts
- top coin contribution chart
- daily portfolio weights
- daily portfolio stats

## How to run

```bash
pip install -r requirements.txt
python src/section7_program.py
