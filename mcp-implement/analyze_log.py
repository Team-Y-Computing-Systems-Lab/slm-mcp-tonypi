import json

# path = "logs/deepseekr1-taskC-robot_trials.jsonl"
path = "logs/robot_trials.jsonl"

total = 0
json_valid = 0
plan_compliant = 0
tool_comp_sum = 0.0
manual_labeled = 0
manual_success = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        total += 1
        if r.get("json_valid"): json_valid += 1
        comp = r.get("compliance", {})
        if comp.get("plan_compliant"): plan_compliant += 1
        tool_comp_sum += float(r.get("tool_compliance_rate", 0.0))

        ms = r.get("manual_success", None)
        if ms is not None:
            manual_labeled += 1
            if ms: manual_success += 1

print("Trials:", total)
print("JSON validity rate:", json_valid / max(1,total))
print("Plan compliance rate:", plan_compliant / max(1,total))
print("Avg tool compliance rate:", tool_comp_sum / max(1,total))
if manual_labeled:
    print("Manual task success rate:", manual_success / manual_labeled, f"(labeled={manual_labeled})")
