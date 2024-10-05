from dotenv import load_dotenv
import streamlit as st
import os
import nltk
nltk.download('punkt')  # Download the tokenizer
nltk.download('stopwords') 
nltk.download('punkt_tab')
import google.generativeai as genai
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
import heapq


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

# Extractive Summarization Function
def extractive_summary(text, num_sentences=3):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    # Filter out stop words and non-alphabetic tokens
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Get frequency distribution of words
    word_freq = FreqDist(words)
    
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    # Score sentences based on word frequency
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] += word_freq[word]
    
    # Get the top n sentences as the summary
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Initialize Streamlit app
st.set_page_config(page_title="CHATBOT DEMO")
st.header("Chatbot Application")

input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# If ask button is clicked
if submit:
    response = get_gemini_response(input_text)
    full_response = "".join([chunk.text for chunk in response])

    # Display the original response
    st.subheader("The Response is")
    st.write(full_response)

    # Generate and display the extractive summary
    summary = extractive_summary(full_response, num_sentences=3)
    st.subheader("Summary:")
    st.write(summary)

    # Display conversation history
    st.write(chat.history)
