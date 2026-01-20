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

# Load config to determine if huggingface authentication is needed
config = FactChecker._load_config("config.yaml")
if not config:
    logger.error("Error loading config: config.yaml is empty or could not be loaded")
    st.error("Error loading config: config.yaml is empty or could not be loaded")
    st.stop()
deployment_type = config.get("deployment_type", "inference_client")
wiki_agentic_rag = config.get("wiki_agentic_rag", False)
needs_auth = deployment_type == "inference_client" or wiki_agentic_rag

# Perform huggingface authentication if needed
if needs_auth:
    # Try to get HF_TOKEN from Streamlit secrets first, then environment variables
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception as e:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN is missing from both secrets.toml and environment variables!")
            st.error("HF_TOKEN is missing from both secrets.toml and environment variables!")
            st.stop()
    #Login to Hugging Face Hub
    try:
        login(token=hf_token)
        logger.info("Logged in to Hugging Face Hub!")
    except Exception as e:
        logger.error(f"Error logging in to Hugging Face Hub: {e}")
        st.error(f"Error logging in to Hugging Face Hub: {e}")
        st.stop()
else:
    logger.info(f"Hugging Face authentication not needed.")

try:
    fact_checker = FactChecker(config_path="config.yaml")
except Exception as e:
    logger.error(f"Error loading FactChecker: {e}")
    st.error(f"Error loading FactChecker: {e}")
    st.stop()

#Receive input from user
fact_check_input = st.text_area("Enter the text to be fact checked:", height=100, placeholder="e.g., The moon is made of cheese.")

#If user clicks check fact button
if st.button("Check Fact"):
    if not fact_check_input:
        st.warning("Please enter a text to be fact checked.")
    else:
        logger.info(f"Fact check input: {fact_check_input}")
        with st.spinner("Analyzing claim..."):
            # Call fact checker check_fact method
            fact_check_result = fact_checker.check_fact(fact_check_input)
            # Display result in a structured manner
            if fact_check_result:
                st.divider()
                st.subheader("Fact Check Result:")
                # Display true or false
                if fact_check_result['is_fact_true']:
                    st.success("**Fact is true**")
                else:
                    st.error("**Fact is false**")
                # Display reasoning 
                with st.expander("View Reasoning", expanded=True):
                    st.markdown(f"*{fact_check_result['reasoning']}*")
            else:
                st.error("**No fact check result found**")