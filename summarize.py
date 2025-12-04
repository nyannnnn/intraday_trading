import re
from datetime import date

LOG_PATH = "logs/paper_trade.log"

# Use the date string that appears in your log timestamps
# If your log lines start with "2025-12-04 ...", this works:
today_str = date.today().strftime("%Y-%m-%d")
# Or hardcode if you want to summarize a specific day:
# today_str = "2025-12-04"

exit_pattern = re.compile(r"\[EXIT\].*pnl=([-0-9.]+)")

trades = []

with open(LOG_PATH, "r") as f:
    for line in f:
        # Only keep lines from today's date
        if today_str not in line:
            continue
        if "[EXIT]" not in line:
            continue

        m = exit_pattern.search(line)
        if not m:
            continue
        pnl = float(m.group(1))
        trades.append(pnl)

n_trades = len(trades)
wins = [p for p in trades if p > 0]
losses = [p for p in trades if p < 0]
flats = [p for p in trades if p == 0]

total_realized = sum(trades)
avg_pnl = total_realized / n_trades if n_trades > 0 else 0.0
win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
max_win = max(wins) if wins else 0.0
max_loss = min(losses) if losses else 0.0

print(f"=== SUMMARY FOR {today_str} (from log) ===")
print(f"Trades: {n_trades}, wins={len(wins)}, losses={len(losses)}, flats={len(flats)}")
print(f"Total realized PnL: {total_realized:.2f}")
print(f"Avg PnL/trade: {avg_pnl:.2f}")
print(f"Win rate: {win_rate:.1%}")
print(f"Max win: {max_win:.2f}, max loss: {max_loss:.2f}")
