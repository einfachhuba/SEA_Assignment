import streamlit as st

def css_blocks():
    st.markdown("""
    <style>
    .profile-card { display:flex; gap:16px; align-items:flex-start;
        border:1px solid rgba(255,255,255,.15); border-radius:12px; padding:14px 16px;
        background: rgba(255,255,255,.03); }
    .profile-avatar { width:84px; height:84px; border-radius:10px; object-fit:cover;
        border:1px solid rgba(255,255,255,.15); }
    .profile-title { font-weight:700; font-size:1.05rem; margin:0; }
    .profile-sub { color:rgba(255,255,255,.7); margin:2px 0 8px 0; font-size:.9rem; }
    .profile-meta { font-size:.85rem; color:rgba(255,255,255,.8); }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px;
        background:rgba(127,90,240,.2); color:#cbb5ff; font-size:.75rem; margin-right:6px; }
    .linkrow a { margin-right:12px; }
    .joke-card{ border:1px solid rgba(255,255,255,.15); background: rgba(255,255,255,.04);
        border-radius:12px; padding:14px 16px; margin:6px 0 18px 0; }
    .joke-text{ margin:0; font-size:0.95rem; line-height:1.4; color:rgba(255,255,255,.9); }
    .pb-wrap { height:10px; background:rgba(255,255,255,.12); border-radius:999px; overflow:hidden; }
    .pb-fill { height:10px; transition: width .3s ease; border-radius:999px; }
    </style>
    """, unsafe_allow_html=True)

def progress_bar(value: float):
    pct = max(0, min(1, value))
    color = "#2cb67d" if pct >= 1.0 else "#4f8bf9"
    st.markdown(f"""
    <div class="pb-wrap"><div class="pb-fill" style="width:{pct*100:.0f}%; background:{color};"></div></div>
    """, unsafe_allow_html=True)
    st.caption(f"{pct*100:.0f}%")

def spacer(px=16):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)