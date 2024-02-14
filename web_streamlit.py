import streamlit as st
from bit_agent_interface import *



# print(final_response)

st.title("ChatGPT-like clone")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = f"{run_graph(prompt)}"
    
    with st.chat_message("assistant"):
        
         st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})

