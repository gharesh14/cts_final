"""Quick checker to verify three ML models load and can predict.

This imports the already-initialized models and helpers from app.py
without starting the Flask server (guarded by __main__). It prints the
status of each service-specific model and runs a test prediction for
each using simple synthetic inputs.
"""

from typing import Dict, Any

from app import (
    models,
    scalers,
    ML_MODEL_AVAILABLE,
    predict_appeal_risk,
    predict_appeal_level,
)


def print_status() -> None:
    print("ML_MODEL_AVAILABLE:", ML_MODEL_AVAILABLE)
    for k, m in models.items():
        print(f"Model[{k}]:", "loaded" if m is not None else "missing", "| Scaler:", "yes" if scalers.get(k) is not None else "no")


def sample_payload(service_type: str) -> Dict[str, Dict[str, Any]]:
    patient = {
        "age": 55,
        "gender": "M",
        "state": "CA",
        "risk_score": 0.6,
    }
    request = {
        "prior_denials": 1,
        "deductible": 1000.0,
        "coinsurance": 20.0,
    }

    if service_type == "medication":
        service = {
            "service_type": "medication",
            "service_name": "Atorvastatin",
            "service_code": "E78.5",
            "tier": 2,
            "requires_pa": False,
            "is_high_cost": False,
            "step_therapy": False,
            "estimated_cost": 120.0,
        }
    elif service_type == "procedure":
        service = {
            "service_type": "procedure",
            "service_name": "Knee Arthroscopy",
            "service_code": "M17.9",
            "tier": 4,
            "requires_pa": True,
            "is_high_cost": True,
            "step_therapy": False,
            "estimated_cost": 7500.0,
        }
    else:  # dme
        service = {
            "service_type": "dme",
            "service_name": "CPAP Device",
            "service_code": "G47.33",
            "tier": 3,
            "requires_pa": True,
            "is_high_cost": False,
            "step_therapy": False,
            "estimated_cost": 850.0,
        }

    return {"service": service, "patient": patient, "request": request}


def run_checks() -> None:
    print_status()
    for key in ["medication", "procedure", "dme"]:
        print("\n===", key.upper(), "===")
        if models.get(key) is None:
            print(f"Skipping {key}: model not loaded")
            continue
        payload = sample_payload(key)
        try:
            should_appeal, confidence, risk_level = predict_appeal_risk(
                payload["service"], payload["patient"], payload["request"]
            )
            level_pct = predict_appeal_level(
                payload["service"], payload["patient"], payload["request"]
            )
            print(
                f"Prediction -> should_appeal={should_appeal}, confidence={confidence:.3f}, risk={risk_level}, level={level_pct:.2f}%"
            )
        except Exception as e:
            print(f"Error running prediction for {key}: {e}")


if __name__ == "__main__":
    run_checks()
