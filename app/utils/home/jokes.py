import requests

def fetch_dad_joke(timeout: int = 10) -> str:
    headers = {"Accept": "application/json", "User-Agent": "SEA-Streamlit-Dashboard"}
    r = requests.get("https://icanhazdadjoke.com/", headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("joke", "No joke today 😅")