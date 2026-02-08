import requests
import json

ROBOT_BASE_URL = "http://100.71.207.226:9030"

def test_robot_commands():
    commands = [
        {
            "name": "Wave",
            "data": {
                "jsonrpc": "2.0",
                "method": "RunAction",
                "params": ["Wave", 1],
                "id": 1
            }
        }
    ],
    {
        "name": "Stand",
        "data": {
                "jsonrpc": "2.0",
                "method": "RunAction",
                "params": ["Stand", 1],
                "id": 1
        }
    }
    
    for cmd in commands:
        try:
            print(f"Testing: {cmd['name']}")
            response = requests.post(ROBOT_BASE_URL, json=cmd['data'], timeout=10)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Error with {cmd['name']}: {e}")
            #response = requests.post(ROBOT_BASE_URL, json=cmd['data'], timeout=10)
            #print(f"Status: {response.status_code}")
            #print(f"Response: {response.text}")

if __name__ == "__main__":
    test_robot_commands()