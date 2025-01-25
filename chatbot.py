import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from transformers import pipeline

# Load knowledge base
knowledge_base_file = "knowledge_base.csv"
data = pd.read_csv(knowledge_base_file)

# Encode knowledge base using Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
knowledge_embeddings = model.encode(data['content'].tolist())

# Create a FAISS index for semantic search
dimension = knowledge_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(knowledge_embeddings)

# Function for retrieving relevant documents
def retrieve_relevant_documents(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    return data.iloc[indices[0]].to_dict('records')

# Set up lightweight generative model
generator = pipeline('text-generation', model='distilgpt2', device=-1)  # CPU-based generation

# Troubleshooting function
def troubleshoot_compressor(issue_description):
    relevant_docs = retrieve_relevant_documents(issue_description)
    context = " ".join([doc['content'] for doc in relevant_docs])
    prompt = f"Compressor troubleshooting based on the issue '{issue_description}': {context}"
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Streamlit app for chatbot interface
st.title("Compressor Troubleshooting AI Chatbot")

# Text input to ask the chatbot for troubleshooting advice
user_issue = st.text_input("Describe the compressor issue:")

# Button to get response from chatbot
if st.button("Get Troubleshooting Advice"):
    if user_issue:
        st.write("Processing your query...")
        response = troubleshoot_compressor(user_issue)
        st.subheader("AI Generated Troubleshooting Advice:")
        st.write(response)
    else:
        st.write("Please enter a compressor issue description.")

