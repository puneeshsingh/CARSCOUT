import streamlit as st

from ingest import SHORTLIST
from retrieve import DEFAULT_MIN_SCORE, search_complaints

st.set_page_config(page_title="CarScout Retrieval Debug", layout="wide")
st.title("CarScout — Retrieval Debug Tool")
st.caption("Retrieval-only. No LLM calls.")

SHORTLIST_OPTIONS = [(make.title(), model.title(), year) for make, model, year, _ in SHORTLIST]
OPTION_LABELS = [
    f"{make} {model}" + (f" ({year})" if year else " (any year)") for make, model, year in SHORTLIST_OPTIONS
]

selected_label = st.selectbox("Make / Model (shortlist)", OPTION_LABELS)
default_make, default_model, default_year = SHORTLIST_OPTIONS[OPTION_LABELS.index(selected_label)]

col1, col2, col3 = st.columns(3)
with col1:
    make = st.text_input("Make", value=default_make)
with col2:
    model = st.text_input("Model", value=default_model)
with col3:
    year = st.number_input("Year", min_value=1990, max_value=2030, value=default_year or 2016, step=1)

query = st.text_input("Query", value="engine stalling")
top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
min_score = st.slider("Min similarity score", min_value=0.0, max_value=1.0, value=DEFAULT_MIN_SCORE, step=0.01)

if st.button("Search", type="primary"):
    with st.spinner("Searching..."):
        try:
            response = search_complaints(make, model, int(year), query, top_k=top_k, min_score=min_score)
        except Exception as e:
            st.error(f"Search failed: {e}")
            response = None

    if response is not None:
        if response.status == "no_confident_match":
            st.warning(response.message)
        else:
            st.success(response.message)
            for r in response.results:
                header = f"#{r.complaint_id} — score {r.score:.4f} — {r.make} {r.model} {r.year}"
                with st.expander(header):
                    st.write(r.narrative)
