#!/usr/bin/env python3
"""Simple CLI for the Gemini REST API."""
import os
import sys
import requests

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def main(prompt: str) -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY environment variable not set")

    url = f"{API_URL}?key={api_key}"
    body = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    response = requests.post(url, json=body)
    response.raise_for_status()
    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        print(data)
    else:
        print(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gemini_interface.py 'your prompt'")
        raise SystemExit(1)
    prompt = " ".join(sys.argv[1:])
    main(prompt)

