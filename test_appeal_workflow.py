#!/usr/bin/env python3
"""
Test script to verify the appeal workflow between doctor and admin dashboards.
"""

import requests
import json
import time

# Base URL for the application
BASE_URL = "http://localhost:5000"

def test_appeal_workflow():
    """Test the complete appeal workflow."""
    print("🧪 Testing Appeal Workflow")
    print("=" * 50)
    
    # Step 1: Check if doctor appeals endpoint works
    print("\n1. Testing Doctor Appeals Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/doctor-appeals")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Doctor appeals endpoint working - Found {len(data.get('appeals', []))} appeals")
        else:
            print(f"❌ Doctor appeals endpoint failed - Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing doctor appeals endpoint: {e}")
    
    # Step 2: Check if admin appeals endpoint works
    print("\n2. Testing Admin Appeals Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/admin-appeals")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Admin appeals endpoint working - Found {len(data.get('appeals', []))} appeals")
        else:
            print(f"❌ Admin appeals endpoint failed - Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing admin appeals endpoint: {e}")
    
    # Step 3: Test appeal submission (if we have a service request)
    print("\n3. Testing Appeal Submission...")
    try:
        # First, get some requests to see if we can submit an appeal
        response = requests.get(f"{BASE_URL}/api/doctor-requests")
        if response.status_code == 200:
            data = response.json()
            requests_list = data.get('requests', [])
            print(f"Found {len(requests_list)} requests")
            
            # Look for a denied request to appeal
            denied_requests = [r for r in requests_list if r.get('overallStatus') == 'Denied']
            if denied_requests:
                print(f"Found {len(denied_requests)} denied requests")
                # Try to submit an appeal for the first denied request
                test_request = denied_requests[0]
                item_id = test_request.get('itemId')
                if item_id:
                    appeal_data = {
                        "item_id": item_id,
                        "appeal_notes": "Test appeal submission",
                        "appeal_documents": "Test documentation"
                    }
                    
                    response = requests.post(f"{BASE_URL}/appeals", json=appeal_data)
                    if response.status_code == 200:
                        print("✅ Appeal submission successful")
                        appeal_result = response.json()
                        print(f"Appeal ID: {appeal_result.get('appeal_id')}")
                    else:
                        print(f"❌ Appeal submission failed - Status: {response.status_code}")
                        print(f"Response: {response.text}")
                else:
                    print("❌ No item_id found in denied request")
            else:
                print("ℹ️ No denied requests found to test appeal submission")
        else:
            print(f"❌ Failed to get requests - Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing appeal submission: {e}")
    
    # Step 4: Test admin appeal decision
    print("\n4. Testing Admin Appeal Decision...")
    try:
        # Get appeals from admin endpoint
        response = requests.get(f"{BASE_URL}/api/admin-appeals")
        if response.status_code == 200:
            data = response.json()
            appeals = data.get('appeals', [])
            print(f"Found {len(appeals)} appeals in admin dashboard")
            
            # Look for a pending appeal to test decision
            pending_appeals = [a for a in appeals if a.get('appeal_outcome') == 'Pending']
            if pending_appeals:
                print(f"Found {len(pending_appeals)} pending appeals")
                test_appeal = pending_appeals[0]
                appeal_id = test_appeal.get('appeal_id')
                
                if appeal_id:
                    # Test approving the appeal
                    decision_data = {
                        "decision": "Approved",
                        "admin_notes": "Test approval by admin"
                    }
                    
                    response = requests.post(f"{BASE_URL}/api/admin-appeals/{appeal_id}/decision", json=decision_data)
                    if response.status_code == 200:
                        print("✅ Admin appeal decision successful")
                        decision_result = response.json()
                        print(f"Decision: {decision_result.get('decision')}")
                    else:
                        print(f"❌ Admin appeal decision failed - Status: {response.status_code}")
                        print(f"Response: {response.text}")
                else:
                    print("❌ No appeal_id found in pending appeal")
            else:
                print("ℹ️ No pending appeals found to test admin decision")
        else:
            print(f"❌ Failed to get admin appeals - Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing admin appeal decision: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Appeal Workflow Test Complete")

if __name__ == "__main__":
    test_appeal_workflow()
