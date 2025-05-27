from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_qa_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
