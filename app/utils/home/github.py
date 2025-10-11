import requests
from functools import lru_cache

API = "https://api.github.com"

def _headers(token: str | None = None):
    return {"Authorization": f"Bearer {token}"} if token else {}

def http_get(path: str, params: dict | None = None, token: str | None = None, timeout: int = 20):
    r = requests.get(f"{API}{path}", params=params, headers=_headers(token), timeout=timeout)
    r.raise_for_status()
    return r.json()

def safe_commits(owner_repo: str, per_page: int = 6, branch: str | None = None, token: str | None = None):
    try:
        params = {"per_page": per_page}
        if branch: params["sha"] = branch
        return http_get(f"/repos/{owner_repo}/commits", params, token)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 409:
            return None  # empty repo
        raise

def open_issues(owner_repo: str, per_page: int = 5, token: str | None = None):
    items = http_get(f"/repos/{owner_repo}/issues", {"state":"open","per_page":per_page}, token)
    return [i for i in items if "pull_request" not in i]

def open_prs(owner_repo: str, per_page: int = 5, token: str | None = None):
    return http_get(f"/repos/{owner_repo}/pulls", {"state":"open","per_page":per_page}, token)

def user(username: str, token: str | None = None):
    return http_get(f"/users/{username}", None, token)