#!/usr/bin/env python3
import csv, json, re
from pathlib import Path

root = Path("results/mcs_specialists/history")
out = Path("results/mcs_specialists/training_summary.csv")
pat = re.compile(r"t(?P<table>\d+)_mcs(?P<mcs>\d+)_(?P<mod>qpsk|16qam|64qam)_(?P<arch>gt_ep|detr_ep)_lmmseH_estR_.*\.json$")
rows = []

for p in sorted(root.glob("*.json")):
    m = pat.match(p.name)
    if not m:
        continue
    hist = json.loads(p.read_text())
    valid = [x for x in hist if x.get("ber_neural") is not None and x.get("ber_ep5") is not None]
    if not valid:
        continue
    init = valid[0]
    best = min(valid, key=lambda x: float(x["ber_neural"]))
    ep5 = float(init["ber_ep5"])
    best_ber = float(best["ber_neural"])
    rows.append({
        "table": int(m.group("table")),
        "mcs": int(m.group("mcs")),
        "modulation": m.group("mod"),
        "arch": m.group("arch"),
        "val_snr_db": float(init.get("val_snr_db", init.get("args", {}).get("val_snr_db", float("nan")))),
        "ep5_ber": ep5,
        "best_neural_ber": best_ber,
        "relative_gain_pct": 100.0 * (ep5 - best_ber) / max(ep5, 1e-12),
        "best_step": int(best["step"]),
        "trueH_ep5_ber": float(init.get("ber_trueH_ep5", float("nan"))),
        "oracle_gap_recovered_pct": 100.0 * float(best.get("oracle_gap_recovered", 0.0)),
    })

rows.sort(key=lambda x: (x["table"], x["mcs"], x["arch"]))
out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

print("="*122)
print("MCS SPECIALIST TRAINING SUMMARY")
print("="*122)
print(f"{'T':>2} {'MCS':>4} {'Mod':>6} {'Arch':>8} {'ValSNR':>7} {'EP5 BER':>11} {'Best BER':>11} {'Gain':>9} {'Step':>6} {'TrueH':>11} {'Recover':>9}")
print("-"*122)
for r in rows:
    print(f"{r['table']:>2} {r['mcs']:>4} {r['modulation']:>6} {r['arch']:>8} {r['val_snr_db']:>+7.2f} {r['ep5_ber']:>11.6e} {r['best_neural_ber']:>11.6e} {r['relative_gain_pct']:>+8.2f}% {r['best_step']:>6} {r['trueH_ep5_ber']:>11.6e} {r['oracle_gap_recovered_pct']:>+8.2f}%")
print("-"*122)
print(f"Saved: {out}")
