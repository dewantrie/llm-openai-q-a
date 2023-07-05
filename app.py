import os
import streamlit as st
import pandas as pd

from user_agent import UserAgent
from helpdesk_agent import HelpdeskAgent

ORGANIZATION = os.environ['OPENAI_ORGANIZATION']
API_KEY = os.environ['OPENAI_API_KEY']
GPT_MODEL = "gpt-3.5-turbo"

dataset_path = "./data/data.csv"
df = pd.read_csv(dataset_path, delimiter=';')
data = df[['description', 'price', 'slug']]

cars = []
for index, row in data.iterrows():
    cars.append({
        'description': row['description'],
        'price': row['price'],
        'slug': row['slug']
    })

st.subheader("AI Assistant")
user_input = st.text_input("You: ", placeholder="Ask me anything ...", key="input")

if st.button("Submit", type="primary"):
    st.markdown("----")
    res_box = st.empty()

    user_agent = UserAgent(ORGANIZATION, API_KEY, GPT_MODEL)
    question, answer = user_agent.ask_question(user_input)

    helpdesk_agent = HelpdeskAgent(ORGANIZATION, API_KEY, GPT_MODEL)
    answer = helpdesk_agent.ask_answer(question, answer, cars)

    text = ""
    for out in answer:
        text += out
        res_box.markdown(f'{text}')
 
st.markdown("----")
