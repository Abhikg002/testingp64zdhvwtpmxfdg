# config.py
import os

try:
    import streamlit as st
    AWS_ACCESS_KEY = st.secrets["aws_access_key_id"]
    AWS_SECRET_KEY = st.secrets["aws_secret_access_key"]
    AWS_REGION = st.secrets["aws_region"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
