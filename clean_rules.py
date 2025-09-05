import json
import os
from typing import Any, Dict, List


SOURCE_FILE = "rules_1000.json"
OUTPUT_FILE = "rules_1000_clean.json"


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    keep_condition_keys = {"prior_therapies", "HbA1c", "BMI", "LDL"}
    keep_top_keys = ("rule_id", "id", "service_type", "services", "diagnosis", "decision", "reason")

    for rule in rules:
        new_rule: Dict[str, Any] = {}

        # Copy top-level keys if present and non-empty
        for key in keep_top_keys:
            value = rule.get(key)
            if value not in (None, "", [], {}):
                new_rule[key] = value

        # Filter conditions
        conditions = rule.get("conditions") or {}
        filtered_conditions: Dict[str, Any] = {}
        for key in keep_condition_keys:
            if key in conditions and conditions[key] not in (None, "", [], {}):
                filtered_conditions[key] = conditions[key]
        if filtered_conditions:
            new_rule["conditions"] = filtered_conditions

        kept.append(new_rule)

    return kept


def main() -> None:
    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")
    rules = load_json(SOURCE_FILE)
    cleaned = clean_rules(rules)
    save_json(OUTPUT_FILE, cleaned)
    print(f"Wrote cleaned rules to {OUTPUT_FILE} ({len(cleaned)} entries)")


if __name__ == "__main__":
    main()


