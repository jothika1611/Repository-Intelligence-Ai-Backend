import httpx
import json

def test():
    print("Sending direct request to Ollama...")
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": "Say hello, this is a test.",
                "stream": False
            },
            timeout=30.0
        )
        print("Status:", response.status_code)
        print("Response:", response.json().get("response"))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test()
