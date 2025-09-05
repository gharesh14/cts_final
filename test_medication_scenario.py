#!/usr/bin/env python3
"""
Test script to test the specific medication rule MED-0001.
"""

import json
import sys
import os

# Add the current directory to Python path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the rule engine functions from app.py
from app import evaluate_rules_locally, fallback_appeal_prediction

def test_medication_rule():
    """Test the specific medication rule MED-0001."""
    
    # Load the test scenario
    with open('test_medication_rule.json', 'r') as f:
        test_data = json.load(f)
    
    scenario = test_data['test_scenario']
    input_data = scenario['input']
    
    print("=" * 80)
    print("TESTING MEDICATION RULE MED-0001")
    print("=" * 80)
    print(f"Description: {scenario['description']}")
    print()
    
    # Extract data
    patient = input_data['patient']
    request = input_data['request']
    service = input_data['service']
    
    print("INPUT DATA:")
    print(f"  Patient: {patient['patient_name']} (Risk: {patient['risk_score']}, State: {patient['state']})")
    print(f"  Service: {service['service_name']} (Tier: {service['tier']}, Cost: ${service['estimated_cost']})")
    print(f"  Diagnosis: {request['diagnosis']}")
    print(f"  Prior Therapies: {request['prior_therapies']}")
    print(f"  HbA1c: {request['hba1c']}")
    print(f"  BMI: {request['bmi']}")
    print(f"  PA Required: {service['requires_pa']}, Step Therapy: {service['step_therapy']}")
    print()
    
    # Run rule engine
    print("RUNNING RULE ENGINE...")
    result = evaluate_rules_locally(service, patient, request)
    
    print("DEBUG - Raw return value:")
    print(f"  Result type: {type(result)}, Value: {result}")
    print()
    
    if isinstance(result, tuple) and len(result) == 2:
        status, reason = result
        print("RESULTS:")
        print(f"  Status: {status}")
        print(f"  Reason: {reason}")
        print()
        
        # Test appeal prediction if status is Denied
        if status == "Denied":
            print("APPEAL PREDICTION (since request was denied):")
            appeal_should_appeal, appeal_confidence, appeal_risk = fallback_appeal_prediction(service, patient, request)
            print(f"  Appeal Recommended: {appeal_should_appeal}")
            print(f"  Appeal Confidence: {appeal_confidence:.2f}")
            print(f"  Appeal Risk Level: {appeal_risk}")
        else:
            print("No appeal prediction needed - request was approved.")
    else:
        print("ERROR: Rule engine did not return expected tuple format")
        print(f"Expected: (status, reason), Got: {result}")
    
    print()
    print("=" * 80)
    
    return result

if __name__ == "__main__":
    try:
        test_medication_rule()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

