import requests

# Test if server is reachable
try:
    response = requests.get('http://localhost:5000/')
    print(f"Server response: {response.json()}")
except Exception as e:
    print(f"Server test failed: {e}")