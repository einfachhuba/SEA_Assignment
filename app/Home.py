import streamlit as st, requests
from datetime import datetime
import random

from utils.home.github import user as gh_user, safe_commits, open_issues, open_prs
from utils.home.jokes import fetch_dad_joke
from utils.home.ui import css_blocks, progress_bar, spacer
from utils.home.config import EMOJIS, OWNER_REPO, USERS, TASKS

st.set_page_config(page_title="SEA Dashboard", page_icon="👨🏽‍💻", layout="wide")
css_blocks()
st.markdown(f"# {random.choice(EMOJIS)} SEA – Dashboard")

# --- Team section ---
st.markdown("## 👥 Project Team")

spacer(12)
@st.cache_data(ttl=3600)
def fetch_user_cached(handle: str):
    return gh_user(handle)

cols = st.columns(2, gap="large")
for col, u in zip(cols, USERS):
    try:
        d = fetch_user_cached(u["gh"])
        name = d.get("name") or d["login"]
        handle = d["login"]
        bio = d.get("bio") or ""
        loc = d.get("location") or ""
        comp = d.get("company") or ""
        blog = d.get("blog") or ""
        followers = d.get("followers", 0)
        repos = d.get("public_repos", 0)

        html = f"""
        <div class="profile-card">
          <img class="profile-avatar" src="{d['avatar_url']}" alt="{name}">
          <div>
            <p class="profile-title">{name}</p>
            <p class="profile-sub">@{handle}</p>
            <div class="profile-meta">{'📍 ' + loc + ' · ' if loc else ''}{'🏢 ' + comp if comp else ''}</div>
            <div style="margin:8px 0 10px 0;">
              <span class="badge">⭐ Public Repos: {repos}</span>
              <span class="badge">👥 Followers: {followers}</span>
            </div>
            <div class="profile-meta" style="margin-bottom:8px;">{bio}</div>
            <div class="linkrow">
              <a href="{d['html_url']}" target="_blank">GitHub</a>
              {"<a href='"+u.get("linkedin","")+"' target='_blank'>LinkedIn</a>" if u.get("linkedin") else ""}
            </div>
          </div>
        </div>
        """
        with col:
            st.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        col.error(f"Failed to load @{u['gh']}: {e}")

# --- Joke section ---
spacer(36)
st.markdown("## 😁 Joke of the Day")
@st.cache_data(ttl=3600)
def cached_joke():
    try:
        return fetch_dad_joke()
    except Exception as e:
        return f"No joke today (API error): {e}"
joke = cached_joke()
st.markdown(f"""<div class="joke-card"><p class="joke-text">{joke}</p></div>""", unsafe_allow_html=True)

# --- GitHub project status ---
spacer(36)
st.markdown("## 💾 GitHub Project Status")

@st.cache_data(ttl=300)
def cached_commits(repo: str, n: int = 6): return safe_commits(repo, per_page=n)
@st.cache_data(ttl=300)
def cached_issues(repo: str, n: int = 5):  return open_issues(repo, per_page=n)
@st.cache_data(ttl=300)
def cached_prs(repo: str, n: int = 5):     return open_prs(repo, per_page=n)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.subheader("Latest Commits")
    commits = cached_commits(OWNER_REPO, 6)
    if commits is None:
        st.info("No commits yet – push your first code!")
    elif commits:
        for c in commits:
            msg = c["commit"]["message"].split("\n")[0]
            author = c["commit"]["author"]["name"]
            ts = datetime.fromisoformat(c["commit"]["author"]["date"].replace("Z","+00:00"))
            st.markdown(f"- {msg} — **{author}** ({ts:%Y-%m-%d %H:%M} UTC)")
    else:
        st.write("No commits found.")

with col2:
    st.subheader("Open Issues")
    issues = cached_issues(OWNER_REPO, 5)
    if not issues:
        st.write("No open issues 🎉")
    else:
        for i in issues:
            st.markdown(f"- #{i['number']}: [{i['title']}]({i['html_url']})")

with col3:
    st.subheader("Open Pull Requests")
    prs = cached_prs(OWNER_REPO, 5)
    if not prs:
        st.write("No open PRs 🚀")
    else:
        for p in prs:
            st.markdown(f"- #{p['number']}: [{p['title']}]({p['html_url']})")

# --- Tasks ---
spacer(12)
st.markdown("### 📋 Assignment Progress")
spacer(12)
for t in TASKS:
    with st.container(border=True):
        st.markdown(f"**{t['title']}**")
        progress_bar(t["done"])
        st.page_link(f"pages/{t['page']}", label="Open page")