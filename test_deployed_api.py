import requests
import json
from datetime import datetime
import time
import traceback
import sys

def test_login_security_api():
    """Test the deployed login security API with various scenarios"""
    
    base_url = "https://mindalike-fraud-detection.onrender.com"  # Render deployment URL
    
    # First test if the API is accessible
    try:
        print(f"Attempting to connect to: {base_url}")
        print(f"Python version: {sys.version}")
        print(f"Requests version: {requests.__version__}")
        
        health_check = requests.get(f"{base_url}/", timeout=10)
        print(f"Health Check Status Code: {health_check.status_code}")
        print(f"Health Check Response Headers: {dict(health_check.headers)}")
        print(f"Health Check Response Text: {health_check.text}")
        
        if health_check.status_code not in [200, 404]:
            print(f"API seems to be down. Status code: {health_check.status_code}")
            print(f"Response: {health_check.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to API: {str(e)}")
        print(traceback.format_exc())
        return
        
    print("API is accessible. Starting tests...")
    
    # Test cases
    test_cases = [
        {
            "timestamp": time.time(),
            "user_id": "test_user_1",
            "ip_address": "192.168.1.1",
            "device_type": "desktop",
            "browser": "Chrome",
            "location": "New York",
            "login_success": 1
        },
        {
            "timestamp": time.time(),
            "user_id": "test_user_2",
            "ip_address": "172.16.0.1",
            "device_type": "mobile",
            "browser": "Safari",
            "location": "London",
            "login_success": 0
        }
    ]
    
    print("\nTesting Login Security API...")
    print(f"Base URL: {base_url}")
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(json.dumps(test_case, indent=2))
        
        try:
            # Make request to analyze_login endpoint
            print(f"Sending request to: {base_url}/analyze_login")
            response = requests.post(
                f"{base_url}/analyze_login",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # Print response
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Raw Response: {response.text}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print("\nParsed Response:")
                    print(json.dumps(result, indent=2))
                    
                    # Test getting user patterns
                    print(f"\nFetching patterns for user {test_case['user_id']}...")
                    patterns_response = requests.get(
                        f"{base_url}/user_patterns/{test_case['user_id']}",
                        timeout=10
                    )
                    print(f"Patterns Status Code: {patterns_response.status_code}")
                    if patterns_response.status_code == 200:
                        patterns = patterns_response.json()
                        print("User Patterns:")
                        print(json.dumps(patterns, indent=2))
                    else:
                        print(f"Error fetching patterns: {patterns_response.text}")
                    
                except json.JSONDecodeError:
                    print("Could not parse JSON response")
            else:
                print(f"Error Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {str(e)}")
            print(traceback.format_exc())
        
        # Add delay between requests
        time.sleep(1)

if __name__ == "__main__":
    test_login_security_api()
