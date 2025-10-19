import streamlit as st, requests
from datetime import datetime
import random

from utils.home.github import user as gh_user, safe_commits, open_prs, open_issues_sorted
from utils.home.ui import css_blocks, progress_bar, spacer, get_github_css
from utils.home.jokes import fetch_dad_joke
from utils.home.ui import css_blocks, progress_bar, spacer
from utils.home.config import EMOJIS, OWNER_REPO, USERS, TASKS

st.set_page_config(page_title="SEA Dashboard", page_icon="👨🏽‍💻", layout="wide")
css_blocks()
st.markdown(get_github_css(), unsafe_allow_html=True)
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

# --- Chat Interface Call-to-Action ---
spacer(12)
# Add custom CSS for blue button
st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #0066CC !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #0052A3 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🤖 Try our AI Chat Interface", use_container_width=True, type="primary"):
        st.switch_page("pages/02_Chat_Interface.py")

# --- GitHub project status ---
spacer(36)
st.markdown("## 💾 GitHub Project Status")

@st.cache_data(ttl=300)
def cached_commits(repo: str, n: int = 100):
    return safe_commits(repo, per_page=n)
@st.cache_data(ttl=300)
def cached_issues_oldest(repo: str, n: int = 100):
    return open_issues_sorted(repo, per_page=n, oldest_first=True, state="all")
@st.cache_data(ttl=300)
def cached_prs(repo: str, n: int = 5):
    return open_prs(repo, per_page=n)


# --- Commits ---
st.markdown("## Latest Commits")
commits = cached_commits(OWNER_REPO, 100)
if commits is None:
    st.info("No commits yet – push your first code!")
elif commits:
    html = ["<div class='gh-card'>",
            "<div class='gh-table-header'><div class='gh-col'>Commit</div><div class='gh-col gh-col-right'>Committer</div></div>"]
    html.append("<div style='max-height:320px;overflow-y:auto;'>")
    for c in commits:
        msg = c["commit"]["message"].split("\n")[0]
        author = c["commit"]["author"].get("name") if c["commit"]["author"] else None
        author = author or (c.get("author") or {}).get("login") or "Unknown"
        date_iso = c["commit"]["author"]["date"]
        ts = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
        date_tag = f"<span class='gh-tag gh-tag-green'>{ts:%Y-%m-%d %H:%M} UTC</span>"
        left = f"<span class='gh-commit-msg'>{msg}</span>{date_tag}"
        avatar = (c.get("author") or {}).get("avatar_url")
        right = ""
        if avatar:
            right += f"<img class='gh-avatar' src='{avatar}' alt='{author}'> "
        right += f"<div class='gh-committer'><strong>{author}</strong></div>"
        html.append(f"<div class='gh-row-bg'><div class='gh-col'>{left}</div><div class='gh-col-right'>{right}</div></div>")
    html.append("</div></div>")
    st.markdown('\n'.join(html), unsafe_allow_html=True)
else:
    st.write("No commits found.")

# --- Issues ---
st.markdown("## Open Issues")
issues = cached_issues_oldest(OWNER_REPO, 100)
if not issues:
    st.write("No open issues")
else:
    html = ["<div class='gh-card'>",
            "<div class='gh-table-header'><div class='gh-col'>Issue</div><div class='gh-col gh-col-right'>Last changed by</div></div>"]
    html.append("<div style='max-height:320px;overflow-y:auto;'>")
    for i in issues:
        title = i.get('title')
        url = i.get('html_url')
        labels = i.get('labels', [])
        # Type tag
        type_tag = ''
        if labels:
            first = labels[0]
            lname = (first.get('name') or '').lower()
            if 'bug' in lname:
                type_cls = 'gh-label-bug'
            elif 'doc' in lname or 'docs' in lname:
                type_cls = 'gh-label-doc'
            else:
                type_cls = 'gh-label-feature'
            type_tag = f"<span class='gh-label {type_cls}'>{first.get('name')}</span>"
        # Status tag
        state = i.get('state')
        if state == 'open':
            status_tag = "<span class='gh-tag gh-tag-red'>Open</span>"
        else:
            status_tag = "<span class='gh-tag gh-tag-green'>Closed</span>"
        left = f"<a class='gh-link' href='{url}' target='_blank'>#{i.get('number')}: {title}</a> {type_tag} {status_tag}"
        # Last changed by
        who = (i.get('closed_by') or i.get('user') or {}).get('login') or 'unknown'
        right = f"<div class='gh-meta'>Last changed by: {who}</div>"
        html.append(f"<div class='gh-row-bg'><div class='gh-col'>{left}</div><div class='gh-col-right'>{right}</div></div>")
    html.append("</div></div>")
    st.markdown('\n'.join(html), unsafe_allow_html=True)

# --- Pull Requests ---
st.markdown("## Open Pull Requests")
prs = cached_prs(OWNER_REPO, 5)
if not prs:
    st.write("No open PRs 🚀")
else:
    html = ["<div class='gh-card'>",
            "<div class='gh-table-header'><div class='gh-col'>Pull Request</div><div class='gh-col gh-col-right'>Author / State</div></div>"]
    for p in prs:
        title = p.get('title')
        url = p.get('html_url')
        num = p.get('number')
        user = (p.get('user') or {}).get('login') or 'unknown'
        state = p.get('state')
        state_cls = 'gh-pr-open' if state == 'open' else 'gh-pr-closed'
        left = f"<a class='gh-link' href='{url}' target='_blank'>#{num}: {title}</a>"
        right = f"<div class='gh-meta'>{user}</div><span class='gh-pr-state {state_cls}'>{state}</span>"
        html.append(f"<div class='gh-row-bg'><div class='gh-col'>{left}</div><div class='gh-col-right'>{right}</div></div>")
    html.append("</div>")
    st.markdown('\n'.join(html), unsafe_allow_html=True)

# --- Tasks ---
spacer(12)
st.markdown("### 📋 Assignment Progress")
spacer(12)
for t in TASKS:
    with st.container(border=True):
        st.markdown(f"**{t['title']}**")
        progress_bar(t["done"])
        st.page_link(f"pages/{t['page']}", label="Open page")