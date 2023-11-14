# Import necessary libraries
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# Set up page configuration
st.set_page_config(page_title="VascularGPT", layout="wide")

# Pinecone index name from the environment variable or Streamlit secrets
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX", "vasculargpt")

# Sidebar configuration
st.sidebar.title("VascularGPT")
st.sidebar.markdown("Ask your vascular-related questions!")

# Main page layout
st.title("Welcome to VascularGPT")
st.markdown("Your AI assistant for vascular knowledge.")

# Initialize Pinecone retriever
vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})  # Adjust 'k' as needed

# Initialize LangChain ConversationalRetrievalChain
conv_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(),
    retriever=retriever,
    return_source_documents=True
)

# Initialize an empty chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input for question
query_input = st.text_input("Enter your question here:", key="query_input")

# Button to submit the query
submit_button = st.button("Submit", key="submit_button")

# Process query
if submit_button and query_input:
    st.write("Processing your question...")  # Removed key parameter
    
    # Append the new query to the chat history
    st.session_state.chat_history.append((query_input, ''))  # Tuple format
    
    # Execute the query against the ConversationalRetrievalChain
    response = conv_chain({
        'question': query_input, 
        'chat_history': st.session_state.chat_history
    })

    # Extract the answer
    answer = response.get('answer', "No answer found.")

    # Handle source documents
    source_documents = response.get('source_documents', [])
    top_source_document = "No source document found."
    if source_documents:
        # Assuming source_documents is a list of Document objects
        top_source_document = getattr(source_documents[0], 'page_content', top_source_document)

    # Update the last entry in chat history with the response
    st.session_state.chat_history[-1] = (query_input, answer)

    # Display the response and the top source document
    st.write("Answer:", answer)  # Removed key parameter
    st.write("Source (Top Document):", top_source_document)  # Removed key parameter
