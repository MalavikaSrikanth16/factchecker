import streamlit as st
from fact_checker import FactChecker

fact_check_input = st.chat_input("Enter text that needs to be fact checked : ")

if fact_check_input:
    # Initialize FactChecker from config file
    fact_checker = FactChecker()
    result = fact_checker.check_fact(fact_check_input)
    print(result)
    if result:
        st.write(result)
    else:
        st.write("No result found")