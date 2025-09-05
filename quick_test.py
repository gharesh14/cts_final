#!/usr/bin/env python3
"""
Quick test to check if the Flask app is running and endpoints are working.
"""

import requests
import json

def test_endpoints():
    base_url = "http://localhost:5000"
    
    print("Testing Flask app endpoints...")
    
    # Test if app is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ App is running - Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ App is not running - Connection refused")
        return
    except Exception as e:
        print(f"❌ Error connecting to app: {e}")
        return
    
    # Test doctor appeals endpoint
    try:
        response = requests.get(f"{base_url}/api/doctor-appeals", timeout=5)
        print(f"Doctor appeals endpoint - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Found {len(data.get('appeals', []))} appeals")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"❌ Error testing doctor appeals: {e}")
    
    # Test admin appeals endpoint
    try:
        response = requests.get(f"{base_url}/api/admin-appeals", timeout=5)
        print(f"Admin appeals endpoint - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Found {len(data.get('appeals', []))} appeals")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"❌ Error testing admin appeals: {e}")

if __name__ == "__main__":
    test_endpoints()
