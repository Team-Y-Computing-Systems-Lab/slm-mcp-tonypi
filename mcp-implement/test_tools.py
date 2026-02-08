

import os, json 
from datetime import datetime
from typing import Any, Dict, List, Tuple

ALLOWED_TOOLS = {
    "Propagate Action": {"required": ["Action"]},
    "Control Servo": {"required": ["Servo Position"]},
    "Capture Image": {"required": ["Request"], "optional": ["BoundaryColors"]},
    "Summarize Scene": {"required": []},
    "Pick Object": {"required": ["object_description"]},
}

ERROR_KEYWORDS = ["error", "failed", "invalid", "not one of", "validation", "timeout"]

def is_error_text(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in ERROR_KEYWORDS)

def validate_step(step: Dict[str, Any], allowed_actions: List[str]) -> Tuple[bool, List[str]]:
    """Return (is_valid, violations)."""
    violations = []
    tool = step.get("tool", "")
    params = step.get("params", {})

    # tool name
    if tool not in ALLOWED_TOOLS:
        return False, [f"Unknown tool: {tool}"]

    if not isinstance(params, dict):
        return False, [f"Params must be dict for tool {tool}"]

    req = ALLOWED_TOOLS[tool].get("required", [])
    opt = ALLOWED_TOOLS[tool].get("optional", [])

    # required params present
    for k in req:
        if k not in params:
            violations.append(f"Missing param '{k}' for tool {tool}")

    # optionally: reject unexpected params
    allowed_keys = set(req + opt)
    for k in params.keys():
        if k not in allowed_keys:
            violations.append(f"Unexpected param '{k}' for tool {tool}")

    # tool-specific checks
    if tool == "Propagate Action":
        act = params.get("Action", None)
        if act not in allowed_actions:
            violations.append(f"Invalid Action '{act}'")
    elif tool == "Control Servo":
        pos = params.get("Servo Position", None)
        if not isinstance(pos, int):
            violations.append("Servo Position must be int")
        else:
            if pos < 1000 or pos > 2000:
                violations.append("Servo Position out of range [1000,2000]")
    elif tool == "Capture Image":
        reqstr = params.get("Request", "")
        if not isinstance(reqstr, str) or not reqstr.strip():
            violations.append("Request must be a non-empty string")
        if "BoundaryColors" in params and not isinstance(params["BoundaryColors"], str):
            violations.append("BoundaryColors must be a string if provided")
    elif tool == "Pick Object":
        desc = params.get("object_description", "")
        if not isinstance(desc, str) or not desc.strip():
            violations.append("object_description must be non-empty string")
    elif tool == "Summarize Scene":
        # Must have no params (or empty dict)
        if len(params) != 0:
            violations.append("Summarize Scene params must be {}")

    return (len(violations) == 0), violations

def validate_plan(plan_data: Dict[str, Any], allowed_actions: List[str]) -> Dict[str, Any]:
    """Returns compliance metrics + violations."""
    metrics = {
        "parsed_ok": True,
        "num_steps": 0,
        "num_compliant_steps": 0,
        "plan_compliant": False,
        "step_violations": [],  # list of {step_index, violations}
    }

    plan = plan_data.get("plan", [])
    if not isinstance(plan, list):
        metrics["parsed_ok"] = False
        return metrics

    metrics["num_steps"] = len(plan)
    for i, step in enumerate(plan):
        ok, v = validate_step(step, allowed_actions)
        if ok:
            metrics["num_compliant_steps"] += 1
        else:
            metrics["step_violations"].append({"step_index": i, "violations": v})

    metrics["plan_compliant"] = (metrics["num_steps"] > 0 and metrics["num_compliant_steps"] == metrics["num_steps"])
    return metrics

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def verify_compliance(plan_data, list_of_all_actions, log_file_name = "robot_trials"): 
    compliance = validate_plan(plan_data, list_of_all_actions)
    tool_compliance_rate = (compliance["num_compliant_steps"] / max(1, compliance["num_steps"]))
    print()
    print(f"compliance: {compliance}, tool compliance rate {tool_compliance_rate}")
    
    label = input("Task success? (y/n/skip): ").strip().lower()
    manual_success = None
    if label in ("y","yes"): manual_success = True
    elif label in ("n","no"): manual_success = False
    
    # auto_success = (compliance["plan_compliant"] and not any(is_error_text(line) for line in execution_log))
    
    log_path = f"logs/{log_file_name}.jsonl"
    trial = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "user_input": user_input,
        "json_valid": plan_data.get("_json_valid", True),
        "compliance": compliance,
        "tool_compliance_rate": tool_compliance_rate,
        "auto_success": compliance["plan_compliant"],
        "manual_success": manual_success,
        # optionally store plan + summary
        "plan": plan_data.get("plan", []),
        # "execution_summary": execution_result,  # can be large; store if you want
    }
    append_jsonl(log_path, trial)
