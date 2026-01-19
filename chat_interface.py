import streamlit as st
import logging
from fact_checker import FactChecker
from huggingface_hub import login
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#Hugging face authentication for inference client
try:
    hf_token = st.secrets["HF_TOKEN"]
except KeyError:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN is missing from secrets.toml and environment variables!")
        hf_token = None

if hf_token:
    login(token=hf_token)
    logger.info("Logged in to Hugging Face Hub!")
else:
    logger.warning("Hugging Face token not found in secrets.")

# Set page configuration
st.set_page_config(layout="wide", page_title="Fact Check App")
st.title("Fact Checker")

#Receive input from user
fact_check_input = st.text_area("Enter the text to be fact checked:", height=100, placeholder="e.g., The moon is made of cheese.")

if st.button("Check Fact"):
    if not fact_check_input:
        st.warning("Please enter a text to be fact checked.")
    else:
        logger.info(f"Fact check input: {fact_check_input}")
        with st.spinner("Analyzing claim..."):
            # Initialize FactChecker from config file
            fact_checker = FactChecker("config.yaml")
            # Call fact checker 
            fact_check_result = fact_checker.check_fact(fact_check_input)
            logger.info(f"Fact check result: {fact_check_result}")
            #Display result
            if fact_check_result:
                st.divider()
                st.subheader("Fact Check Result:")
                if fact_check_result['is_fact_true']:
                    st.success("**Fact is true**")
                else:
                    st.error("**Fact is false**")
                with st.expander("View Reasoning & Evidence", expanded=True):
                    st.subheader("Reasoning:")
                    st.markdown(f"*{fact_check_result['reasoning']}*")
            else:
                st.error("**No fact check result found**")