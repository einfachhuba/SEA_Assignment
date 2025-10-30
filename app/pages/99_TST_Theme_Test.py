import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

from utils.general.ui import spacer

st.set_page_config(
    page_title="Theme Test",
    page_icon="🎨",
    layout="wide",
)

# Resolve dark mode from session state with a safe default
dark_mode = st.session_state.get("dark_mode", False)

# Apply theme configuration dynamically
if dark_mode:
    # Dark mode theme
    st.markdown("""
    <style>
        :root {
            color-scheme: dark;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light mode theme
    st.markdown("""
    <style>
        :root {
            color-scheme: light;
        }
    </style>
    """, unsafe_allow_html=True)


st.markdown("# 🎨 Theme Test Page")
st.markdown("Test your light/dark mode theme with dummy data")

spacer(24)

# --- Test Containers ---
st.markdown("## Container Test")
with st.container(border=True):
    st.markdown("**Test Container**")
    st.write("This is a test container to verify border colors and background.")
    st.button("Test Button", type="primary")

spacer(24)

# --- Test DataFrames ---
st.markdown("## DataFrame Test with st.dataframe()")

# Create dummy commit data
commits_data = []
authors = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
messages = [
    "Fix bug in authentication",
    "Add new feature for user dashboard",
    "Update documentation",
    "Refactor database queries",
    "Improve error handling",
    "Add unit tests",
    "Update dependencies",
    "Fix typo in README",
]

for i in range(20):
    date = datetime.now() - timedelta(days=i, hours=random.randint(0, 23))
    commits_data.append({
        "Message": random.choice(messages),
        "Author": random.choice(authors),
        "Date": date.strftime("%Y-%m-%d %H:%M CET")
    })

df_commits = pd.DataFrame(commits_data)
st.dataframe(df_commits, width='stretch', height=400)

spacer(24)

# --- Test Data Editor ---
st.markdown("## DataFrame Test with st.data_editor()")

# Create dummy issues data
issues_data = []
statuses = ["open", "closed", "in-progress"]
labels = ["bug", "enhancement", "documentation", "question"]
users = ["user1", "user2", "user3", "user4"]

for i in range(15):
    issue_title = random.choice([
        'Bug fix needed',
        'Feature request',
        'Documentation update',
        'Question about API'
    ])
    issues_data.append({
        "#": i + 1,
        "Title": f"Issue {i + 1}: {issue_title}",
        "Status": random.choice(statuses),
        "Label": random.choice(labels),
        "Opened By": random.choice(users),
        "URL": f"https://github.com/example/repo/issues/{i + 1}"
    })

df_issues = pd.DataFrame(issues_data)
st.data_editor(
    df_issues,
    width='stretch',
    hide_index=True,
    height=400,
    disabled=True,
    column_config={
        "URL": st.column_config.LinkColumn("Link"),
        "#": st.column_config.NumberColumn("#", width="small"),
        "Status": st.column_config.TextColumn("Status", width="small"),
        "Label": st.column_config.TextColumn("Label", width="medium")
    },
    key="test_issues_table"
)

spacer(24)

# --- Test Pull Requests ---
st.markdown("## Pull Requests Test")

pr_data = []
pr_statuses = ["open", "merged", "closed"]
pr_users = ["dev1", "dev2", "dev3"]

for i in range(10):
    pr_title = random.choice([
        'Add feature',
        'Fix bug',
        'Update docs',
        'Refactor code'
    ])
    pr_data.append({
        "#": i + 100,
        "Title": f"PR {i + 100}: {pr_title}",
        "Status": random.choice(pr_statuses),
        "Requester": random.choice(pr_users),
        "URL": f"https://github.com/example/repo/pull/{i + 100}"
    })

df_prs = pd.DataFrame(pr_data)
st.data_editor(
    df_prs,
    width='stretch',
    hide_index=True,
    height=300,
    disabled=True,
    column_config={
        "URL": st.column_config.LinkColumn("Link"),
        "#": st.column_config.NumberColumn("#", width="small"),
        "Status": st.column_config.TextColumn("Status", width="small")
    },
    key="test_prs_table"
)

spacer(24)

# --- Test Text Elements ---
st.markdown("## Text Elements Test")
with st.container(border=True):
    st.markdown("### Heading 3")
    st.markdown("**Bold Text**")
    st.markdown("*Italic Text*")
    st.write("Regular paragraph text")
    st.caption("Caption text")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Metric Label", "1,234")
    with col2:
        st.metric("Another Metric", "5,678")

spacer(24)

# --- Test Input Elements ---
st.markdown("## Input Elements Test")
with st.container(border=True):
    st.text_input("Text Input", "Sample text")
    st.selectbox("Select Box", ["Option 1", "Option 2", "Option 3"])
    st.checkbox("Checkbox")
    st.button("Button", type="primary")

spacer(24)

st.success("✅ Theme test page loaded successfully!")
st.info("ℹ️ Toggle Dark Mode above to test both themes")
