import streamlit as st
from datetime import datetime
import pandas as pd
import pytz

from utils.home.github import user as gh_user, safe_commits, all_prs, open_issues_sorted
from utils.home.ui import css_blocks, progress_bar
from utils.home.jokes import fetch_dad_joke
from utils.home.config import OWNER_REPO, USERS, TASKS
from utils.general.ui import spacer

st.set_page_config(page_title="SEA Dashboard", layout="wide")
css_blocks()
st.markdown("# SEA - Dashboard")

# --- Team section ---
st.markdown("## Project Team")

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
        followers = d.get("followers", 0)
        repos = d.get("public_repos", 0)
        job_title = u.get("job_title", "")

        # Build card content using native Streamlit
        with col:
            with st.container(border=True):
                # Avatar and name section
                col_avatar, col_info = st.columns([1, 3])
                with col_avatar:
                    st.image(d['avatar_url'], width=100)
                with col_info:
                    st.markdown(f"### {name}")
                    st.caption(f"@{handle}")
                    # Build meta line: Location · Job @ Company
                    meta_parts = []
                    if loc:
                        meta_parts.append(f"📍 {loc}")
                    if job_title and comp:
                        meta_parts.append(f"🏢 {job_title} @ {comp}")
                    elif job_title:
                        meta_parts.append(f"🏢 {job_title}")
                    elif comp:
                        meta_parts.append(f"🏢 {comp}")
                    if meta_parts:
                        st.caption(" · ".join(meta_parts))
                
                # Stats
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Public Repos", repos)
                with stat_col2:
                    st.metric("Followers", followers)
                
                # Bio
                if bio:
                    st.markdown(bio)
                
                # Links
                linkedin_link = u.get("linkedin", "")
                link_text = f"[GitHub]({d['html_url']})"
                if linkedin_link:
                    link_text += f" • [LinkedIn]({linkedin_link})"
                st.markdown(link_text)
    except Exception as e:
        col.error(f"Failed to load @{u['gh']}: {e}")

# --- Joke section ---
spacer(36)
st.markdown("## Joke of the Day")


def cached_joke():
    try:
        return fetch_dad_joke()
    except Exception as e:
        return f"No joke today (API error): {e}"


joke = cached_joke()
with st.container(border=True):
    st.markdown(f"**{joke}**")

# --- GitHub project status ---
spacer(36)
st.markdown("## GitHub Project Status")


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
    st.info("No commits yet - push your first code!")
elif commits:
    rows = []
    ce_tz = pytz.timezone('Europe/Vienna')
    for c in commits:
        msg = (c.get("commit", {}).get("message") or "").split("\n")[0]
        author_name = None
        if c.get("commit", {}).get("author"):
            author_name = c["commit"]["author"].get("name")
            date_iso = c["commit"]["author"].get("date")
        else:
            date_iso = None
        author_login = (c.get("author") or {}).get("login")
        author = author_name or author_login or "Unknown"
        # Convert date
        if date_iso:
            ts = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
            ts_local = ts.astimezone(ce_tz)
            date_str = ts_local.strftime("%Y-%m-%d %H:%M %Z")
        else:
            date_str = ""
        html_url = c.get("html_url")
        sha = (c.get("sha") or "")[:7]
        rows.append({
            "SHA": sha,
            "Message": msg,
            "Committer": author,
            "Date": date_str,
            "URL": html_url,
        })
    df_commits = pd.DataFrame(rows)
    st.data_editor(
        df_commits,
        hide_index=True,
        disabled=True,
        width='stretch',
        height=320,
        column_config={
            "URL": st.column_config.LinkColumn("Link", display_text="Open"),
            "SHA": st.column_config.TextColumn("SHA", width="small"),
            "Message": st.column_config.TextColumn("Message", width="large"),
            "Committer": st.column_config.TextColumn("Committer", width="medium"),
            "Date": st.column_config.TextColumn("Date", width="medium"),
        },
        key="home_commits_table",
    )
else:
    st.write("No commits found.")

# --- Issues ---
st.markdown("## Open Issues")
issues = cached_issues_oldest(OWNER_REPO, 100)
if not issues:
    st.write("No open issues")
else:
    issue_rows = []
    for i in issues:
        labels = i.get('labels', [])
        first_label = labels[0].get('name') if labels else ''
        issue_rows.append({
            "#": i.get('number'),
            "Title": i.get('title'),
            "Status": i.get('state'),
            "Label": first_label,
            "Opened By": (i.get('user') or {}).get('login') or 'unknown',
            "URL": i.get('html_url'),
        })
    df_issues = pd.DataFrame(issue_rows)
    st.data_editor(
        df_issues,
        hide_index=True,
        disabled=True,
        width='stretch',
        height=320,
        column_config={
            "URL": st.column_config.LinkColumn("Link", display_text="Open"),
            "#": st.column_config.NumberColumn("#", width="small"),
            "Title": st.column_config.TextColumn("Title", width="large"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Label": st.column_config.TextColumn("Label", width="medium"),
            "Opened By": st.column_config.TextColumn("Opened By", width="medium"),
        },
        key="home_issues_table",
    )

# --- Pull Requests ---
st.markdown("## Pull Requests")
prs = cached_prs(OWNER_REPO, 10)
if not prs:
    st.write("No PRs found 🚀")
else:
    pr_rows = []
    for p in prs:
        state = p.get('state')
        merged = p.get('merged_at') is not None
        status_text = 'open' if state == 'open' else ('merged' if merged else 'closed')
        pr_rows.append({
            "#": p.get('number'),
            "Title": p.get('title'),
            "Status": status_text,
            "Requester": (p.get('user') or {}).get('login') or 'unknown',
            "URL": p.get('html_url'),
        })
    df_prs = pd.DataFrame(pr_rows)
    st.data_editor(
        df_prs,
        hide_index=True,
        disabled=True,
        width='stretch',
        height=280,
        column_config={
            "URL": st.column_config.LinkColumn("Link", display_text="Open"),
            "#": st.column_config.NumberColumn("#", width="small"),
            "Title": st.column_config.TextColumn("Title", width="large"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Requester": st.column_config.TextColumn("Requester", width="medium"),
        },
        key="home_prs_table",
    )

# --- Tasks ---
spacer(12)
st.markdown("### Assignment Progress")
spacer(12)
for t in TASKS:
    with st.container(border=True):
        st.markdown(f"**{t['title']}**")
        progress_bar(t["done"])
        st.page_link(f"pages/{t['page']}", label="Open page")

# --- AI Chat Interface ---
spacer(24)
st.markdown("### AI Assistant")

with st.container(border=True):
    chat_cols = st.columns([6, 2])
    
    with chat_cols[0]:
        st.markdown(
            "<p class='profile-title' style='margin-bottom: 4px; "
            "margin-top: 10px;'>AI Chat Interface</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='profile-sub' style='margin: 0;'>You have some questions? "
            "Try asking our AI assistant!</p>",
            unsafe_allow_html=True,
        )
    
    with chat_cols[1]:
        spacer(4)
        if st.button("Try Now →", width='stretch', type="primary"):
            st.switch_page("pages/02_Chat_Interface.py")
        spacer(4)
