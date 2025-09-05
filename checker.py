import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# Prefer the consolidated rules_1000.json if present
RULES_FILENAME = "rules_1000.json" if os.path.exists("rules_1000.json") else "uhc_rules.json"


def load_rules(rules_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load rules from JSON file located next to this script by default."""
    if rules_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(script_dir, RULES_FILENAME)

    with open(rules_path, "r", encoding="utf-8") as f:
        rules: List[Dict[str, Any]] = json.load(f)
    return rules


def normalize_service_name(name: str) -> str:
    return (name or "").strip().lower()


def is_diagnosis_match(rule_diagnoses: List[str], diagnosis: str) -> bool:
    # Exact code match; codes are case-insensitive but otherwise exact
    normalized = (diagnosis or "").strip().upper()
    return any(str(code).strip().upper() == normalized for code in (rule_diagnoses or []))


def _parse_threshold(expr: Any) -> Optional[Tuple[str, float]]:
    try:
        text = str(expr).strip()
        if text.startswith(">="):
            return (">=", float(text[2:].strip()))
        if text.startswith(">"):
            return (">", float(text[1:].strip()))
        if text.startswith("<="):
            return ("<=", float(text[2:].strip()))
        if text.startswith("<"):
            return ("<", float(text[1:].strip()))
        return ("==", float(text))
    except Exception:
        return None


def _meets_threshold(value: Any, threshold: Optional[Tuple[str, float]]) -> bool:
    try:
        if value is None or threshold is None:
            return False
        op, tval = threshold
        v = float(value)
        if op == ">":
            return v > tval
        if op == ">=":
            return v >= tval
        if op == "<":
            return v < tval
        if op == "<=":
            return v <= tval
        return v == tval
    except Exception:
        return False


def find_matching_rule(
    rules: List[Dict[str, Any]], service_name: str, diagnosis: str, service_type: Optional[str] = None, request: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    target_service = normalize_service_name(service_name)
    target_type = (service_type or "").strip().lower()

    for rule in rules:
        # New schema allows either explicit services list or service_type
        rule_services = [normalize_service_name(s) for s in (rule.get("services") or [])]
        rule_type = (rule.get("service_type") or "").strip().lower()
        has_service_match = (target_service and target_service in rule_services) or (target_type and rule_type and target_type == rule_type and not rule_services)
        if not has_service_match:
            continue

        # Diagnosis array match if provided
        if rule.get("diagnosis"):
            if not is_diagnosis_match(rule.get("diagnosis") or [], diagnosis):
                continue

        # Evaluate conditions if present
        conditions = rule.get("conditions") or {}
        conditions_ok = True
        if conditions:
            # prior_therapies: all required must be in provided
            required_prior = [str(x).strip().lower() for x in (conditions.get("prior_therapies") or [])]
            if required_prior:
                provided_prior = (request or {}).get("prior_therapies") or []
                provided_prior_norm = [str(x).strip().lower() for x in provided_prior]
                if not all(any(req in p for p in provided_prior_norm) for req in required_prior):
                    conditions_ok = False

            # Numeric thresholds
            for key, req_key in [("HbA1c", "hba1c"), ("BMI", "bmi"), ("LDL", "ldl")]:
                if key in conditions:
                    thr = _parse_threshold(conditions.get(key))
                    if not _meets_threshold((request or {}).get(req_key), thr):
                        conditions_ok = False
                        break

        if not conditions_ok:
            continue

        return rule

    return None


def to_binary_decision(decision_raw: str) -> str:
    # Map to Approved/Denied only
    text = (decision_raw or "").strip().lower()
    if "denied" in text:
        return "Denied"
    return "Approved"


def check_coverage(
    service_name: str, diagnosis: str, rules_path: Optional[str] = None, service_type: Optional[str] = None, request: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (binary_decision, details_dict)

    details_dict contains: rule_id, decision_raw, reason, tier, service_type
    """
    rules = load_rules(rules_path)
    rule = find_matching_rule(rules, service_name, diagnosis, service_type, request)
    if not rule:
        details = {
            "rule_id": None,
            "decision_raw": "No matching rule",
            "reason": "No rule matched service and diagnosis",
            "tier": None,
            "service_type": service_type,
        }
        return "Denied", details

    decision_raw = rule.get("decision", "")
    binary = to_binary_decision(decision_raw)
    details = {
        "rule_id": rule.get("rule_id") or rule.get("id"),
        "decision_raw": decision_raw,
        "reason": rule.get("reason"),
        "tier": rule.get("tier"),
        "service_type": rule.get("service_type") or service_type,
    }
    return binary, details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check UHC rules: returns Approved/Denied for {service_name, diagnosis}."
    )
    parser.add_argument("--service", required=True, help="Service name, e.g., Atorvastatin")
    parser.add_argument("--dx", required=True, help="Diagnosis code (ICD-10), e.g., E78.5")
    parser.add_argument(
        "--rules",
        required=False,
        default=None,
        help="Path to rules JSON (defaults to rules_1000.json or uhc_rules.json)",
    )
    parser.add_argument(
        "--type",
        required=False,
        default=None,
        help="Optional service_type (e.g., medication, procedure, dme)",
    )
    parser.add_argument(
        "--prior_therapies",
        required=False,
        default=None,
        help="Comma-separated list of prior therapies",
    )
    parser.add_argument("--hba1c", required=False, default=None, type=float)
    parser.add_argument("--bmi", required=False, default=None, type=float)
    parser.add_argument("--ldl", required=False, default=None, type=float)

    args = parser.parse_args()

    req_payload: Dict[str, Any] = {}
    if args.prior_therapies:
        req_payload["prior_therapies"] = [x.strip() for x in args.prior_therapies.split(",") if x.strip()]
    if args.hba1c is not None:
        req_payload["hba1c"] = args.hba1c
    if args.bmi is not None:
        req_payload["bmi"] = args.bmi
    if args.ldl is not None:
        req_payload["ldl"] = args.ldl

    decision, details = check_coverage(
        args.service,
        args.dx,
        args.rules,
        service_type=args.type,
        request=req_payload if req_payload else None,
    )
    output = {
        "service_name": args.service,
        "diagnosis": args.dx,
        "decision": decision,  # Approved/Denied
        "details": details,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


