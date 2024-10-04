import os
import xml.etree.ElementTree as ET
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Load environment variables
load_dotenv()


# Function to load MedQuAD dataset from folders
def load_medquad_data(base_dir):
    qa_pairs = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_path.endswith('.xml'):  # Assuming the files are in XML format
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    for qa in root.findall(".//QAPair"):  # Adjust based on the structure of the XML
                        question = qa.find("Question").text
                        answer = qa.find("Answer").text
                        qa_pairs.append((question, answer))
    return qa_pairs

# Function to perform medical entity recognition
def extract_medical_entities(text):
    medical_terms = {'diabetes', 'hypertension', 'asthma', 'cancer', 'virus', 'infection','symptoms','diseases','treatments','remedy','fever','cold','cough'}  # Define some medical terms
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    
    entities = [(word, 'MEDICAL_TERM') for word, tag in tagged_tokens if word.lower() in medical_terms]
    return entities

# Function to find the most relevant answer using TF-IDF and cosine similarity
def find_relevant_answer(user_question, qa_pairs, threshold=0.3):
    questions = [qa[0] for qa in qa_pairs]  # Extract all the questions
    tfidf_vectorizer = TfidfVectorizer().fit_transform(questions + [user_question])
    
    # Compute cosine similarity between user question and dataset questions
    cosine_sim = cosine_similarity(tfidf_vectorizer[-1], tfidf_vectorizer[:-1]).flatten()
    
    # Get the index of the most similar question
    most_similar_idx = cosine_sim.argmax()
    similarity_score = cosine_sim[most_similar_idx]
    
    # Only return an answer if the similarity score is above the threshold
    if similarity_score >= threshold:
        return qa_pairs[most_similar_idx][1], questions[most_similar_idx], similarity_score
    else:
        return None, None, similarity_score

# Load MedQuAD data (update 'base_dir' with the correct path to your MedQuAD dataset)
base_dir = "d:\\MedQuAD-master"
qa_pairs = load_medquad_data(base_dir)

# Streamlit app
st.title("Medical Q&A Chatbot")
st.write("Ask your medical questions below:")

# Input question from user
user_question = st.text_input("Your Question:")

# Button to submit question and process response
if st.button("Ask"):
    if user_question:
        # Extract medical entities from the user question
        medical_entities = extract_medical_entities(user_question)
        st.write("**Recognized Medical Entities:**")
        if medical_entities:
            for entity, label in medical_entities:
                st.write(f"- {entity} ({label})")
        else:
            st.write("No medical entities recognized.")
        
        # Find the most relevant answer using TF-IDF and cosine similarity
        answer, matched_question, similarity_score = find_relevant_answer(user_question, qa_pairs)

        if answer:
            st.write(f"**Matched Question:** {matched_question}")
            st.write(f"**Answer:** {answer}")
            st.write(f"**Similarity Score:** {similarity_score:.2f}")

            # Store the conversation in session state
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            st.session_state['history'].append((user_question, answer))
        else:
            st.write("Sorry, no relevant answer found for your question.")
    else:
        st.write("Please enter a question.")

# Display conversation history
if 'history' in st.session_state:
    st.write("**Conversation History:**")
    for idx, (q, a) in enumerate(st.session_state['history']):
        st.write(f"Q{idx+1}: {q}")
        st.write(f"A{idx+1}: {a}")