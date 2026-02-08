import json, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math

LOG_ROOT = Path("logs")

# Adjust this regex to match your filenames
# Examples supported:
#   deepseek-taskC-robot_trials.jsonl
#   qwen-taskA.jsonl
#   smollm2-taskE.jsonl
FILENAME_RE = re.compile(r"(?P<model>[a-zA-Z0-9_]+)-task(?P<task>[A-Za-z])", re.IGNORECASE)

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + z**2/(4*n))/n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

def summarize_jsonl(path: Path):
    total = 0
    json_valid = 0
    plan_compliant = 0
    tool_comp_sum = 0.0

    manual_labeled = 0
    manual_success = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            if r.get("json_valid"):
                json_valid += 1

            comp = r.get("compliance", {})
            if comp.get("plan_compliant"):
                plan_compliant += 1

            tool_comp_sum += float(r.get("tool_compliance_rate", 0.0))

            ms = r.get("manual_success", None)
            if ms is not None:
                manual_labeled += 1
                if ms:
                    manual_success += 1

    return {
        "trials": total,
        "json_valid_rate": json_valid / total if total else 0.0,
        "plan_compliant_rate": plan_compliant / total if total else 0.0,
        "avg_tool_compliance": tool_comp_sum / total if total else 0.0,
        "manual_labeled": manual_labeled,
        "manual_success_rate": (manual_success / manual_labeled) if manual_labeled else None,
        "manual_success_k": manual_success,
    }

rows = []
for p in LOG_ROOT.rglob("*.jsonl"):
    m = FILENAME_RE.search(p.name)
    if not m:
        continue
    model = m.group("model").lower()
    task = m.group("task").upper()

    s = summarize_jsonl(p)
    s.update({"model": model, "task": f"Task {task}", "file": str(p)})
    rows.append(s)

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No files matched. Adjust FILENAME_RE or folder structure.")

# Sort tasks A..Z
df["task_sort"] = df["task"].str.extract(r"Task ([A-Z])")[0]
df = df.sort_values(["task_sort", "model"]).drop(columns=["task_sort"])

print(df[["task","model","trials","json_valid_rate","plan_compliant_rate","avg_tool_compliance","manual_labeled","manual_success_rate"]])

# ---------- Plot 1: Manual task success rate (where available) ----------
# Use manual success if available; else leave missing (NaN)
plot_df = df.copy()
plot_df["success_rate"] = plot_df["manual_success_rate"]

# Keep only tasks/models where manual success exists (usually C/D/E)
plot_df = plot_df.dropna(subset=["success_rate"])

if not plot_df.empty:
    pivot = plot_df.pivot(index="task", columns="model", values="success_rate")
    ax = pivot.plot(kind="bar", figsize=(9, 4), rot=0)
    ax.set_ylabel("Manual Task Success Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("fig_task_success_manual.png", dpi=300)
    plt.show()

# ---------- Plot 2: Plan compliance rate ----------
pivot2 = df.pivot(index="task", columns="model", values="plan_compliant_rate")
ax2 = pivot2.plot(kind="bar", figsize=(9, 4), rot=0)
ax2.set_ylabel("Plan Compliance Rate")
ax2.set_xlabel("")
ax2.set_ylim(0, 1.0)
ax2.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig_plan_compliance.png", dpi=300)
plt.show()

# ---------- Plot 3 (optional): Avg tool compliance ----------
pivot3 = df.pivot(index="task", columns="model", values="avg_tool_compliance")
ax3 = pivot3.plot(kind="bar", figsize=(9, 4), rot=0)
ax3.set_ylabel("Average Tool Compliance (Call-level)")
ax3.set_xlabel("")
ax3.set_ylim(0, 1.0)
ax3.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig_tool_compliance.png", dpi=300)
plt.show()

# Save a summary table for the paper
df.to_csv("summary_metrics.csv", index=False)
print("Saved: fig_task_success_manual.png, fig_plan_compliance.png, fig_tool_compliance.png, summary_metrics.csv")
