import streamlit as st, requests
from datetime import datetime
import random

from utils.home.github import user as gh_user, safe_commits, all_prs, open_issues_sorted
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
    return all_prs(repo, per_page=n)


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
            "<div class='gh-table-header'><div class='gh-col'>Issue</div><div class='gh-col gh-col-right'>Opened By</div></div>"]
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
            status_tag = "<span class='gh-tag gh-tag-red'>open</span>"
        else:
            status_tag = "<span class='gh-tag gh-tag-green'>closed</span>"
        left = f"<a class='gh-link' href='{url}' target='_blank'>#{i.get('number')}: {title}</a> {type_tag} {status_tag}"
        
        # Opened by with avatar like commits
        user_data = i.get('user') or {}
        opener = user_data.get('login') or 'unknown'
        avatar = user_data.get('avatar_url')
        
        right = ""
        if avatar:
            right += f"<img class='gh-avatar' src='{avatar}' alt='{opener}'> "
        right += f"<div class='gh-committer'><strong>{opener}</strong></div>"
        
        html.append(f"<div class='gh-row-bg'><div class='gh-col'>{left}</div><div class='gh-col-right'>{right}</div></div>")
    html.append("</div></div>")
    st.markdown('\n'.join(html), unsafe_allow_html=True)

# --- Pull Requests ---
st.markdown("## Pull Requests")
prs = cached_prs(OWNER_REPO, 10)
if not prs:
    st.write("No PRs found 🚀")
else:
    html = ["<div class='gh-card'>",
            "<div class='gh-table-header'><div class='gh-col'>Pull Request</div><div class='gh-col gh-col-right'>Requester</div></div>"]
    for p in prs:
        title = p.get('title')
        url = p.get('html_url')
        num = p.get('number')
        user_data = p.get('user') or {}
        user = user_data.get('login') or 'unknown'
        avatar = user_data.get('avatar_url')
        state = p.get('state')
        merged = p.get('merged_at') is not None
        
        # Determine state class and display text
        if state == 'open':
            state_cls = 'gh-pr-open'
            state_text = 'open'
        elif merged:
            state_cls = 'gh-pr-merged'
            state_text = 'merged'
        else:
            state_cls = 'gh-pr-closed'
            state_text = 'closed'
            
        left = f"<a class='gh-link' href='{url}' target='_blank'>#{num}: {title}</a> <span class='gh-pr-state {state_cls}'>{state_text}</span>"
        
        # Build right side with avatar and username like commits
        right = ""
        if avatar:
            right += f"<img class='gh-avatar' src='{avatar}' alt='{user}'> "
        right += f"<div class='gh-committer'><strong>{user}</strong></div>"
        
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

# --- AI Chat Interface ---
spacer(24)
st.markdown("### 🧠 AI Assistant")

with st.container(border=True):
    chat_cols = st.columns([6, 2])
    
    with chat_cols[0]:
        st.markdown("<p class='profile-title' style='margin-bottom: 4px; margin-top: 10px;'>AI Chat Interface</p>", unsafe_allow_html=True)
        st.markdown("<p class='profile-sub' style='margin: 0;'>You have some questions? Try asking our AI assistant!</p>", unsafe_allow_html=True)
    
    with chat_cols[1]:
        spacer(4)
        if st.button("Try Now →", use_container_width=True, type="primary"):
            st.switch_page("pages/02_Chat_Interface.py")
        spacer(4)