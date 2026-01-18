import streamlit as st
from fact_checker import FactChecker

#Receive input from user using streamlit chat input
fact_check_input = st.chat_input("Enter text that needs to be fact checked : ")

if fact_check_input:
    # Initialize FactChecker from config file
    fact_checker = FactChecker()
    # Call fact checker and get result
    result = fact_checker.check_fact(fact_check_input)

    print(result)

    if result:
        st.write(result)
    else:
        st.write("No result found")