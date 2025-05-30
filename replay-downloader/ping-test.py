import requests

# Replace with your actual API key
API_KEY = "PfRoIALT3dfYslK0n0sKsGhar3Dh9zqVUaW7gwyW"
API_URL = "https://ballchasing.com/api/"

headers = {
    "Authorization": API_KEY
}

def ping_ballchasing_api():
    try:
        response = requests.get(API_URL, headers=headers)
        if response.status_code == 200:
            print("✅ API is reachable and the key is valid.")
            print("Response:", response.json())
        elif response.status_code == 401:
            print("❌ Unauthorized: Invalid or missing API key.")
        elif response.status_code == 429:
            print("⚠️ Rate limit hit. Try again later.")
        elif response.status_code == 500:
            print("❌ Server error on Ballchasing side.")
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
            print(response.text)
    except requests.RequestException as e:
        print("❌ Error connecting to Ballchasing API:", str(e))

# Run the function
ping_ballchasing_api()
