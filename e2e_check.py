import json
import os
from typing import Any, Dict, List

from app import run_local_rules_engine, predict_appeal_risk, predict_appeal_level


def load_rules(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_samples(rules: List[Dict[str, Any]], max_per_type: int = 1) -> List[Dict[str, Any]]:
    seen = {"medication": 0, "imaging": 0, "procedure": 0, "dme": 0}
    samples: List[Dict[str, Any]] = []
    for r in rules:
        st = (r.get("service_type") or "").lower()
        if st not in seen:
            continue
        if seen[st] >= max_per_type:
            continue
        services = r.get("services") or []
        diagnosis = (r.get("diagnosis") or [None])[0]
        if not services or not diagnosis:
            continue
        samples.append({
            "service_type": st,
            "service_name": services[0],
            "diagnosis": diagnosis,
            "rule": r,
        })
        seen[st] += 1
        if all(v >= max_per_type for v in seen.values()):
            break
    return samples


def build_payload(sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    service_type = sample["service_type"]
    service_name = sample["service_name"]
    # Reasonable defaults; model feature builder tolerates missing fields
    patient = {"age": 55, "gender": "M", "state": "CA", "risk_score": 0.6}
    request = {"diagnosis": sample["diagnosis"], "prior_denials": 1, "deductible": 1000.0, "coinsurance": 20.0}
    # Heuristic defaults depending on type
    if service_type == "medication":
        tier, requires_pa, is_high_cost, step_therapy, estimated_cost = 2, False, False, False, 120.0
    elif service_type == "imaging":
        tier, requires_pa, is_high_cost, step_therapy, estimated_cost = 3, True, True, False, 2000.0
    elif service_type == "procedure":
        tier, requires_pa, is_high_cost, step_therapy, estimated_cost = 4, True, True, False, 7500.0
    else:  # dme
        tier, requires_pa, is_high_cost, step_therapy, estimated_cost = 3, True, False, False, 850.0

    service = {
        "service_type": service_type,
        "service_name": service_name,
        "tier": tier,
        "requires_pa": requires_pa,
        "is_high_cost": is_high_cost,
        "step_therapy": step_therapy,
        "estimated_cost": estimated_cost,
    }

    return {"service": service, "patient": patient, "request": request}


def main() -> None:
    rules_path = None
    for cand in ["rules_1000.json", "rules_1000_clean.json", "rules.json"]:
        if os.path.exists(cand):
            rules_path = cand
            break
    if not rules_path:
        print("No rules file found.")
        return

    rules = load_rules(rules_path)
    samples = pick_samples(rules, max_per_type=1)
    print(f"Using rules from {rules_path}; picked {len(samples)} samples: {[s['service_type'] for s in samples]}")

    for s in samples:
        payload = build_payload(s)
        service = payload["service"]
        patient = payload["patient"]
        request = payload["request"]

        print("\n===", service["service_type"].upper(), "::", service["service_name"], "===")
        decision, reason = run_local_rules_engine(service, patient, request)
        print("Rule decision:", decision)
        print("Reason:", reason)

        if decision == "Denied":
            should_appeal, confidence, risk = predict_appeal_risk(service, patient, request)
            level = predict_appeal_level(service, patient, request)
            print(f"Appeal -> should_appeal={should_appeal}, confidence={confidence:.3f}, risk={risk}, level={level:.2f}%")
        else:
            print("Appeal not evaluated (decision is Approved)")


if __name__ == "__main__":
    main()





