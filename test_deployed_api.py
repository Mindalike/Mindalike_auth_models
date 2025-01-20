import requests
import json
from datetime import datetime
import time

def test_login_security_api():
    """Test the deployed login security API with various scenarios"""
    
    base_url = "https://mindalike-fraud-detection.onrender.com"  # Render deployment URL
    
    # First test if the API is accessible
    try:
        health_check = requests.get(f"{base_url}/docs")
        if health_check.status_code != 200:
            print(f"API seems to be down. Status code: {health_check.status_code}")
            print(f"Response: {health_check.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to API: {str(e)}")
        return
        
    print("API is accessible. Starting tests...")
    
    # Test cases
    test_cases = [
        {
            "user_id": "test_user_1",
            "timestamp": time.time(),
            "ip_address": "192.168.1.1",
            "device_type": "desktop",
            "browser": "Chrome",
            "location": "New York",
            "login_success": True
        },
        {
            "user_id": "test_user_1",
            "timestamp": time.time(),
            "ip_address": "10.0.0.1",  # Different IP
            "device_type": "mobile",    # Different device
            "browser": "Safari",        # Different browser
            "location": "London",       # Different location
            "login_success": True
        },
        {
            "user_id": "test_user_2",
            "timestamp": time.time(),
            "ip_address": "172.16.0.1",
            "device_type": "desktop",
            "browser": "Firefox",
            "location": "Tokyo",
            "login_success": False
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
        
        # Add delay between requests
        time.sleep(1)

if __name__ == "__main__":
    test_login_security_api()
