import httpx
import time

BASE_URL = "http://localhost:8000/api/v1"

def wait_for_server():
    print("Waiting for server to become available...")
    for _ in range(30):
        try:
            r = httpx.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                print("Server is ready!")
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def run_tests():
    if not wait_for_server():
        print("Error: Server did not start in time.")
        return

    print("\n--- Testing Health API ---")
    r = httpx.get(f"{BASE_URL}/health")
    print(r.status_code, r.json())
    
    print("\n--- Testing DB View API (Queries) ---")
    r = httpx.get(f"{BASE_URL}/db/queries")
    print(r.status_code, r.json())

    print("\n--- Testing DB View API (Responses) ---")
    r = httpx.get(f"{BASE_URL}/db/responses")
    print(r.status_code, r.json())

    print("\n--- Testing Question Endpoint ---")
    body = {"query": "What is the capital of France?", "prepare_tts": False}
    # Increased timeout to 120 seconds to allow LLMs to process
    r = httpx.post(f"{BASE_URL}/ask-question", json=body, timeout=120.0) 
    print(r.status_code, r.text)

    print("\n--- Testing Evaluation Endpoint ---")
    # Increased timeout to 120 seconds here as well
    r = httpx.post(f"{BASE_URL}/evaluate", json={"sample_size": 1}, timeout=120.0) 
    print(r.status_code, r.text)

if __name__ == "__main__":
    run_tests()