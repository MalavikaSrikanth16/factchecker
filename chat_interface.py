import streamlit as st
import logging
import os
from typing import Dict, Any, Optional
from fact_checker import FactChecker
from huggingface_hub import login
from constants import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page layout and title
st.set_page_config(layout="wide", page_title="Fact Check App")
st.title("Fact Checker")


@st.cache_resource
def authenticate_huggingface() -> None:
    """Authenticate and Login to Hugging Face Hub if required based on the configuration.
    
    This function is cached using st.cache_resource to ensure authentication
    only happens once. (until app restart)
    """
    config = FactChecker.load_config(CONFIG_PATH)
    deployment_type = config.get(CONFIG_KEY_DEPLOYMENT_TYPE, CONFIG_INFERENCE_CLIENT_DEPLOYMENT_TYPE)
    wiki_agentic_rag = config.get(CONFIG_KEY_WIKI_AGENTIC_RAG, False)
    needs_auth = deployment_type == CONFIG_INFERENCE_CLIENT_DEPLOYMENT_TYPE or wiki_agentic_rag

    if not needs_auth:
        return

    # Get HF_TOKEN from Streamlit secrets first, then environment variables
    try:
        hf_token = st.secrets[HUGGINGFACE_TOKEN]
    except (KeyError, AttributeError):
        hf_token = os.getenv(HUGGINGFACE_TOKEN)
        if not hf_token:
            error_msg = f"{HUGGINGFACE_TOKEN} is missing from both secrets.toml and environment variables!"
            logger.error(error_msg)
            st.error(error_msg)
            st.stop()

    # Login to Hugging Face Hub
    try:
        login(token=hf_token)
        logger.info("Logged in to Hugging Face Hub!")
    except Exception as e:
        error_msg = f"Error logging in to Hugging Face Hub: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()

@st.cache_resource
def initialize_fact_checker() -> FactChecker:
    """Initialize and return the FactChecker instance.
    
    This function is cached using st.cache_resource to ensure the LLM within FactChecker
    is loaded only once. (until app restart)
    """
    logger.info(f"Initializing FactChecker")
    try:
        return FactChecker(config_path=CONFIG_PATH)
    except Exception as e:
        error_msg = f"Error loading FactChecker: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()

def display_fact_check_result(result: Optional[Dict[str, Any]]) -> None:
    """Display the fact check result in a structured manner."""
    if not result:
        st.error("**No fact check result found**")
        return

    st.divider()
    st.subheader("Fact Check Result:")

    if result['is_fact_true']:
        st.success("**Fact is true**")
    else:
        st.error("**Fact is false**")

    with st.expander("View Reasoning", expanded=True):
        st.markdown(f"*{result['reasoning']}*")

# Authenticate Hugging Face if needed and initialize FactChecker
authenticate_huggingface()
fact_checker = initialize_fact_checker()

# User input
fact_check_input = st.text_area(
    "Enter the text to be fact checked:",
    height=100,
    placeholder="e.g., The moon is made of cheese."
)

# Process fact check request when user clicks the button
if st.button("Check Fact"):
    if not fact_check_input:
        st.warning("Please enter a text to be fact checked.")
    else:
        logger.info(f"Fact check input: {fact_check_input}")
        with st.spinner("Analyzing claim..."):
            fact_check_result = fact_checker.check_fact(fact_check_input)
            display_fact_check_result(fact_check_result)