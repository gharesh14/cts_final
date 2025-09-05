#!/usr/bin/env python3
"""
Test script to verify rule engine fixes against sample input/output scenarios.
"""

import json
import sys
import os

# Add the current directory to Python path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the rule engine functions from app.py
from app import local_rules, fallback_appeal_prediction, _normalize_service_name

def load_sample_data():
    """Load sample input and output data."""
    with open('sample_inputs_doctor_dashboard.json', 'r') as f:
        sample_inputs = json.load(f)
    
    with open('sample_outputs_doctor_dashboard.json', 'r') as f:
        sample_outputs = json.load(f)
    
    return sample_inputs, sample_outputs

def test_scenario(scenario_input, expected_output):
    """Test a single scenario against expected output."""
    scenario_id = scenario_input.get('scenario_id', 'UNKNOWN')
    description = scenario_input.get('description', 'No description')
    
    print(f"\n{'='*60}")
    print(f"Testing {scenario_id}: {description}")
    print(f"{'='*60}")
    
    # Extract data from input
    patient = scenario_input['input']['patient']
    request = scenario_input['input']['request']
    service = scenario_input['input']['service']
    
    # Run rule engine
    status, reason = local_rules(service, patient, request)
    
    # Get expected results
    expected = expected_output['expected_output']
    expected_status = expected['approval_status']
    expected_reason = expected['reason']
    
    # Test appeal prediction if status is Denied
    appeal_should_appeal = False
    appeal_confidence = 0.0
    appeal_risk = None
    
    if status == "Denied":
        appeal_should_appeal, appeal_confidence, appeal_risk = fallback_appeal_prediction(service, patient, request)
    
    # Compare results
    status_match = status == expected_status
    reason_match = reason == expected_reason
    
    print(f"Input:")
    print(f"  Patient: {patient['patient_name']} (Risk: {patient['risk_score']}, State: {patient['state']})")
    print(f"  Service: {service['service_name']} (Tier: {service['tier']}, Cost: ${service['estimated_cost']})")
    print(f"  Diagnosis: {request['diagnosis']}")
    print(f"  PA Required: {service['requires_pa']}, Step Therapy: {service['step_therapy']}")
    
    print(f"\nResults:")
    print(f"  Status: {status} (Expected: {expected_status}) {'✓' if status_match else '✗'}")
    print(f"  Reason: {reason}")
    print(f"  Expected Reason: {expected_reason}")
    print(f"  Reason Match: {'✓' if reason_match else '✗'}")
    
    if status == "Denied":
        print(f"  Appeal Recommended: {appeal_should_appeal}")
        print(f"  Appeal Confidence: {appeal_confidence:.2f}")
        print(f"  Appeal Risk: {appeal_risk}")
        
        # Check appeal prediction against expected
        expected_appeal = expected.get('ml_prediction', {})
        expected_should_appeal = expected_appeal.get('should_appeal', False)
        expected_confidence = expected_appeal.get('confidence', 0.0)
        expected_risk_level = expected_appeal.get('risk_level')
        
        appeal_match = appeal_should_appeal == expected_should_appeal
        confidence_match = abs(appeal_confidence - expected_confidence) < 0.01
        risk_match = appeal_risk == expected_risk_level
        
        print(f"  Expected Appeal: {expected_should_appeal} {'✓' if appeal_match else '✗'}")
        print(f"  Expected Confidence: {expected_confidence:.2f} {'✓' if confidence_match else '✗'}")
        print(f"  Expected Risk: {expected_risk_level} {'✓' if risk_match else '✗'}")
    
    # Overall result
    overall_success = status_match and reason_match
    if status == "Denied":
        overall_success = overall_success and appeal_match and confidence_match and risk_match
    
    print(f"\nOverall Result: {'PASS' if overall_success else 'FAIL'}")
    
    return overall_success

def main():
    """Run all test scenarios."""
    print("Testing Rule Engine Against Sample Scenarios")
    print("=" * 60)
    
    try:
        sample_inputs, sample_outputs = load_sample_data()
    except FileNotFoundError as e:
        print(f"Error loading sample data: {e}")
        return 1
    
    # Create a mapping of scenario_id to expected output
    expected_map = {}
    for scenario in sample_outputs['scenarios']:
        expected_map[scenario['scenario_id']] = scenario
    
    # Test each scenario
    passed = 0
    total = 0
    
    for scenario in sample_inputs['scenarios']:
        scenario_id = scenario['scenario_id']
        if scenario_id in expected_map:
            success = test_scenario(scenario, expected_map[scenario_id])
            if success:
                passed += 1
            total += 1
        else:
            print(f"\nWarning: No expected output found for {scenario_id}")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {passed}/{total} scenarios passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 All tests passed! Rule engine is working correctly.")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Rule engine needs more fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())