import streamlit as st


def css_blocks():
    st.markdown(
        """
    <style>
    .profile-card { 
        display:flex;
        gap:16px;
        align-items:flex-start;
        border:1px solid rgba(255,255,255,.15);
        border-radius:12px;
        padding:14px 16px;
        background: rgba(255,255,255,.03);
        }
    .profile-avatar { 
        width:84px;
        height:84px;
        border-radius:10px;
        object-fit:cover;
        border:1px solid rgba(255,255,255,.15);
        }
    .profile-title {
        font-weight:700;
        font-size:1.05rem;
        margin:0;
        }
    .profile-sub { 
        color:rgba(255,255,255,.7);
        margin:2px 0 8px 0;
        font-size:.9rem;
        }
    .profile-meta { 
        font-size:.85rem;
        color:rgba(255,255,255,.8);
        }
    .badge { 
        display:inline-block;
        padding:2px 8px;
        border-radius:999px;
        background:rgba(127,90,240,.2);
        color:#cbb5ff;
        font-size:.75rem;
        margin-right:6px;
        }
    .linkrow a { 
        margin-right:12px;
        }
    .joke-card{ 
        border:1px solid rgba(255,255,255,.15);
        background: rgba(255,255,255,.04);
        border-radius:12px;
        padding:14px 16px;
        margin:6px 0 18px 0;
        }
    .joke-text{ 
        margin:0;
        font-size:0.95rem;
        line-height:1.4;
        color:rgba(255,255,255,.9);
        }
    .pb-wrap { 
        height:10px;
        background:rgba(255,255,255,.12);
        border-radius:999px;
        overflow:hidden;
        }
    .pb-fill { 
        height:10px;
        transition: width .3s ease;
        border-radius:999px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


# CSS used by the GitHub blocks in the dashboard. Kept in ui.py so all UI
# styling sits in one place and matches the look of the existing blocks.
GITHUB_CSS = """
<style>
/* GitHub card/flex table-like layout */
.gh-card{
    background: rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.15);
    border-radius:12px;
    padding:12px 0 0 0;
    padding-left: 12px;
    padding-right: 12px;
    margin-bottom:18px;
    }
.gh-table-header{
    display:flex;
    align-items:center;
    font-weight:600;
    font-size:1.01rem;
    padding:0 24px 8px 24px;
    color:rgba(255,255,255,.92);
    border-bottom:1px solid rgba(255,255,255,.15);
    margin-bottom:8px;
    letter-spacing:.01em;
}
.gh-table-header .gh-col {
    width: 72%;
    min-width: 0;
    max-width: 72%;
    box-sizing:
    border-box;
    padding-left: 22px;
}
.gh-table-header .gh-col-right {
    width: 28%;
    min-width: 0;
    max-width: 28%;
    box-sizing: border-box;
    text-align: right;
    padding-right: 22px;
}
.gh-row-bg {
    display: flex;
    align-items: center;
    background: rgba(255,255,255,.03);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 8px;
    box-sizing: border-box;
}
.gh-row-bg .gh-col {
    width: 72%;
    min-width: 0;
    max-width: 72%;
    box-sizing:
    border-box;
    align-items:
    center;
    gap: 10px;
    display: flex;
}
.gh-row-bg .gh-col-right {
    width: 28%;
    min-width: 0;
    max-width: 28%;
    box-sizing: border-box;
    justify-content: flex-end;
    text-align: right;
    align-items: center;
    display: flex;
    gap: 10px;
}
.gh-commit-msg,.gh-issue-title,.gh-pr-title{
    font-size:14px;
    color:rgba(255,255,255,.95);
    font-weight:500;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}
.gh-tag{
    display:inline-block;
    padding:4px 8px;
    border-radius:999px;
    font-size:12px;
    color:#fff;
    margin-left:10px;
}
.gh-tag-green{
    background:#16a34a;
}
.gh-label{
    margin-left:8px;
    padding:3px 7px;
    border-radius:6px;
    font-size:12px;
    color:#fff;
}
.gh-label-bug{
    background:#ef4444;
}
.gh-label-feature{
    background:#10b981;
}
.gh-label-doc{
    background:#6366f1;
}
.gh-avatar{
    width:36px;
    height:36px;
    border-radius:8px;
    object-fit:cover;
}
.gh-committer{
    font-size:13px;
    color:rgba(255,255,255,.95);
}
.gh-meta{
    font-size:12px;
    color:rgba(255,255,255,.7);
    margin-left:8px;
}
.gh-pr-state{
    padding:6px 10px;
    border-radius:8px;
    font-size:12px;
    color:#fff;
    margin-left:10px;
}
.gh-pr-open{
    background:#059669;
}
.gh-pr-closed{
    background:#ef4444;
}
.gh-link{
    color:#a5b4fc;
    text-decoration:none;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.gh-row:hover .gh-row-bg{
    box-shadow:0 6px 18px rgba(0,0,0,0.18);
}
</style>
"""


def get_github_css():
    """Return the CSS snippet used for GitHub tables."""
    return GITHUB_CSS


def progress_bar(value: float):
    pct = max(0, min(1, value))
    color = "#2cb67d" if pct >= 1.0 else "#4f8bf9"
    st.markdown(
        f"""
    <div class="pb-wrap"><div class="pb-fill" style="
    width:{pct*100:.0f}%; background:{color};"></div></div>
    """,
        unsafe_allow_html=True,
    )
    st.caption(f"{pct*100:.0f}%")


def spacer(px=16):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)
