"""
app.py
------
Streamlit entry point for FundFinder.

The site itself (search box with live, as-you-type suggestions,
category browser, fund profile pages) lives in fundfinder.html, a
single self-contained HTML/CSS/JS file sitting next to this script.
It stays a plain HTML/JS page -- rather than being rebuilt as
st.text_input + st.button widgets -- because Streamlit reruns the
whole script on every widget interaction, which is too slow/round-
trippy for instant, character-by-character search suggestions. This
file just embeds that finished page via st.components.v1.html and
lets the browser run it exactly as it does standalone.

Folder layout expected:
    app.py            <- this file
    fundfinder.html   <- the website
    requirements.txt

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push both files (plus requirements.txt) to a GitHub repo, then
    point Community Cloud at app.py -- no other setup needed.
"""

import pathlib

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FundFinder", page_icon="📈", layout="wide")

# Streamlit wraps every page in its own chrome (top padding, a hidden
# hamburger menu, a "Made with Streamlit" footer). Hidden here so the
# embedded site can use the full browser window like a normal
# standalone website rather than sitting inside a framed widget.
st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .block-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important;}
      iframe {width: 100%; border: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

HTML_PATH = pathlib.Path(__file__).parent / "fundfinder.html"

try:
    html_code = HTML_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    st.error(
        "Couldn't find fundfinder.html next to app.py. Make sure both "
        "files are in the same folder before running `streamlit run app.py`."
    )
    st.stop()

# height is generous and fixed, with scrolling enabled inside the
# iframe -- the page's own content (category directory, expanded fund
# lists, etc.) scrolls within that frame the same way it would in a
# normal browser tab.
components.html(html_code, height=2400, scrolling=True)
