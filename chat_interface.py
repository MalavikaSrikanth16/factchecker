import streamlit as st
import logging
import os
from fact_checker import FactChecker
from huggingface_hub import login

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page layout and title
st.set_page_config(layout="wide", page_title="Fact Check App")
st.title("Fact Checker")


def authenticate_huggingface():
    """Authenticate with Hugging Face Hub if required by the configuration."""
    config = FactChecker._load_config("config.yaml")
    deployment_type = config.get("deployment_type", "inference_client")
    wiki_agentic_rag = config.get("wiki_agentic_rag", False)
    needs_auth = deployment_type == "inference_client" or wiki_agentic_rag

    if not needs_auth:
        return

    # Get HF_TOKEN from Streamlit secrets first, then environment variables
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except (KeyError, AttributeError):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            error_msg = "HF_TOKEN is missing from both secrets.toml and environment variables!"
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

def initialize_fact_checker():
    """Initialize and return the FactChecker instance."""
    try:
        return FactChecker(config_path="config.yaml")
    except Exception as e:
        error_msg = f"Error loading FactChecker: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()

def display_fact_check_result(result):
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

# Initialize application
authenticate_huggingface()
fact_checker = initialize_fact_checker()

# User input
fact_check_input = st.text_area(
    "Enter the text to be fact checked:",
    height=100,
    placeholder="e.g., The moon is made of cheese."
)

# Process fact check request
if st.button("Check Fact"):
    if not fact_check_input:
        st.warning("Please enter a text to be fact checked.")
    else:
        logger.info(f"Fact check input: {fact_check_input}")
        with st.spinner("Analyzing claim..."):
            fact_check_result = fact_checker.check_fact(fact_check_input)
            display_fact_check_result(fact_check_result)