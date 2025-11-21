import requests


def test_auth_system():
    base_url = "http://localhost:8000"

    print("🚀 Testing JWT Authentication System")
    print("=" * 50)

    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/auth/health")
        print(f"✅ Health check: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print(
            "💡 Make sure to start the server first: python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"
        )
        return

    # Test 2: Register new user
    print("\n2. Testing user registration...")
    user_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "password": "securepass123",
    }

    try:
        response = requests.post(f"{base_url}/auth/register", json=user_data)
        if response.status_code == 201:
            print(f"✅ User registered successfully: {response.json()}")
        else:
            print(f"❌ Registration failed: {response.json()}")
    except Exception as e:
        print(f"❌ Registration error: {e}")

    # Test 3: Login with new user
    print("\n3. Testing user login...")
    login_data = {"email": "john@example.com", "password": "securepass123"}

    try:
        response = requests.post(f"{base_url}/auth/login", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Login successful!")
            print(f"Access Token: {token_data['access_token'][:50]}...")
            print(f"Token Type: {token_data['token_type']}")
            print(f"Expires In: {token_data['expires_in']} seconds")
            return token_data["access_token"]
        else:
            print(f"❌ Login failed: {response.json()}")
    except Exception as e:
        print(f"❌ Login error: {e}")

    return None


if __name__ == "__main__":
    test_auth_system()
