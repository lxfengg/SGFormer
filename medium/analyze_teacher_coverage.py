# 保存为 analyze_teacher_coverage.py
import glob, pandas as pd, os, json, numpy as np

LOG_DIR = "results/epoch_logs"
OUT_JSON = "results/diagnosis_teacher_coverage.json"

files = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))
summary = []

if not files:
    print("[WARN] No epoch log CSVs found in", LOG_DIR)
    print("Put epoch logs under results/epoch_logs/ and re-run.")
    exit(0)

for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"[WARN] Failed to read {f}: {e}")
        continue

    row = {"file": os.path.basename(f), "epochs": len(df)}
    # teacher conf
    if 'mean_teacher_conf' in df.columns:
        mtc_first = float(df['mean_teacher_conf'].iloc[0])
        mtc_last = float(df['mean_teacher_conf'].iloc[-1])
        mtc_mean = float(df['mean_teacher_conf'].mean())
    else:
        mtc_first = mtc_last = mtc_mean = None
    row.update({"mean_teacher_conf_first": mtc_first,
                "mean_teacher_conf_last": mtc_last,
                "mean_teacher_conf_mean": mtc_mean})

    # cons_nodes
    if 'cons_nodes' in df.columns:
        try:
            cn = df['cons_nodes'].astype(float).values
            cn_first = int(cn[0])
            cn_last = int(cn[-1])
            cn_mean = float(np.mean(cn))
        except Exception:
            cn_first = cn_last = cn_mean = None
    else:
        cn_first = cn_last = cn_mean = None
    row.update({"cons_nodes_first": cn_first,
                "cons_nodes_last": cn_last,
                "cons_nodes_mean": cn_mean})

    # val/test trend
    if 'val' in df.columns and 'test' in df.columns:
        val_first = float(df['val'].iloc[0]); val_last = float(df['val'].iloc[-1])
        test_first = float(df['test'].iloc[0]); test_last = float(df['test'].iloc[-1])
        row.update({"val_first": val_first, "val_last": val_last, "val_delta": val_last - val_first,
                    "test_first": test_first, "test_last": test_last, "test_delta": test_last - test_first})
    else:
        row.update({"val_first": None, "val_last": None, "val_delta": None,
                    "test_first": None, "test_last": None, "test_delta": None})

    # small diagnostics / warnings
    notes = []
    if mtc_mean is not None and mtc_mean < 0.4:
        notes.append("LOW_TEACHER_CONF")
    if cn_mean is not None and cn_mean < 1.0:
        notes.append("CONS_NODES_VERY_LOW")
    if row.get("val_delta") is not None and row["val_delta"] < 0.0:
        notes.append("VAL_DECREASE")
    summary.append(row)
    row["notes"] = notes

# write JSON summary
os.makedirs("results", exist_ok=True)
with open(OUT_JSON, "w") as jf:
    json.dump(summary, jf, indent=2)

# print concise one-line summary per file (easy to paste)
for r in summary:
    parts = [
        r["file"],
        f"epochs={r['epochs']}",
        f"mtc_last={r['mean_teacher_conf_last']:.3f}" if r['mean_teacher_conf_last'] is not None else "mtc_last=N/A",
        f"mtc_mean={r['mean_teacher_conf_mean']:.3f}" if r['mean_teacher_conf_mean'] is not None else "mtc_mean=N/A",
        f"cons_last={r['cons_nodes_last']}" if r['cons_nodes_last'] is not None else "cons_last=N/A",
        f"cons_mean={r['cons_nodes_mean']:.1f}" if r['cons_nodes_mean'] is not None else "cons_mean=N/A",
        f"valΔ={r['val_delta']:+.3f}" if r['val_delta'] is not None else "valΔ=N/A",
        f"testΔ={r['test_delta']:+.3f}" if r['test_delta'] is not None else "testΔ=N/A",
    ]
    notes = r.get("notes", [])
    note_str = (" | " + ",".join(notes)) if notes else ""
    print(" | ".join(parts) + note_str)

print()
print(f"JSON summary written to: {OUT_JSON}")
print("Tip: to show JSON summary: jq . results/diagnosis_teacher_coverage.json  (if jq is available)")
