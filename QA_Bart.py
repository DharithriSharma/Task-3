from dotenv import load_dotenv
import streamlit as st
import os
import nltk
import google.generativeai as genai
from transformers import BartTokenizer, BartForConditionalGeneration

# Load environment variables
load_dotenv()

# Initialize Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get Gemini API response
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Initialize BART tokenizer and model
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Abstractive Summarization using BART
def bart_summarization(text, num_beams=4, max_length=150, min_length=40, length_penalty=2.0):
    inputs = bart_tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=num_beams, max_length=max_length, min_length=min_length, length_penalty=length_penalty, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Initialize Streamlit app
st.set_page_config(page_title="CHATBOT DEMO")
st.header("Chatbot Application")

input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# If ask button is clicked
if submit:
    # Get full response from Gemini API
    response = get_gemini_response(input_text)
    full_response = "".join([chunk.text for chunk in response])

    # Display the original response
    st.subheader("The Response is")
    st.write(full_response)

    # Generate and display the BART summary
    summary = bart_summarization(full_response)
    st.subheader("Summary:")
    st.write(summary)

    # Display conversation history
    st.write(chat.history)
