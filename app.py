import os
import time
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from pymongo import MongoClient

load_dotenv()


class EasyMongo:
    """
    A simple wrapper for MongoDB Client.
    """

    def __init__(self):
        """
        Constructor for EasyMongo.
        """
        self.URI = os.getenv("MONGO_URI")
        self.DB = os.getenv("MONGO_DB")
        self.COLLECTION = os.getenv("MONGO_COLLECTION")

    def get_database(self):
        """
        Create a connection to MongoDB Atlas url and return NoSQL Database.
        """
        client = MongoClient(self.URI)

        # Connect the database
        return client[self.DB]

    def get_collection(self):
        """
        Get collection from Database.
        """
        dbname = self.get_database()
        return dbname[self.COLLECTION]

    def insert_many(self, documents: List[Dict]):
        """
        Insert multiple documents to MongoDB.

        :param documents: List of Dictionaries.
        :type documents: List[Dict]
        """
        collection = self.get_collection()

        try:
            # Insert documents
            result = collection.insert_many(documents)
            print(f"Inserted {len(result.inserted_ids)} documents")
            print(f"Inserted document IDs: {result.inserted_ids}")
        except Exception as e:
            print(f"Error: {e}")

    def test_data(self):
        """
        Dummy data to test MongoDB connection.
        """
        user_content = {
            "role": "user",
            "content": "What is machine learning in 200 characters?",
        }
        ai_content = {
            "role": "assistant",
            "content": "Machine learning is a subset of artificial intelligence that "
            "enables computers to learn and improve their performance on a "
            "task without explicitly programmed instructions, by using "
            "algorithms and statistical models to analyze and learn "
            "from data.",
        }
        user_content2 = {
            "role": "user",
            "content": "What is deep learning in 200 characters?",
        }
        ai_content2 = {
            "role": "assistant",
            "content": "Deep learning is a subset of machine learning that utilizes "
            "neural networks with multiple layers to learn and represent "
            "complex patterns in data. It enables AI models to recognize "
            "and make decisions based on intricate relationships within "
            "the data, leading to improved accuracy and efficiency in "
            "various applications such as image recognition, natural "
            "language processing, and speech recognition.",
        }

        self.insert_many([user_content, ai_content, user_content2, ai_content2])


def create_message(role: str, content: str) -> Dict:
    return {LLMStrings.ROLE_ID: role, LLMStrings.CONTENT: content}


def output_text(llm_model: Ollama, text: str) -> str:
    prompt_template = f"{LLMStrings.PROMPT_TEMPLATE} {text}"
    return llm_model(prompt_template)


def simulate_response(text: str):
    message_placeholder = st.empty()
    full_response = ""
    time_delay = 0.05

    for chunk in text.split():
        full_response += chunk + " "
        time.sleep(time_delay)
        message_placeholder.markdown(full_response + "▌")

    message_placeholder.markdown(full_response)


class LLMStrings:
    """
    A brief description of the MyClass class.
    """

    # Q&A strings
    PROMPT_TEMPLATE = """
    In the field of Oncology, what is the answer to the following question?
    """
    GREETINGS = "Welcome to the EAGLE: A Multimodal Medical Q&A Chatbot! Ask me anything related to Oncology."
    WAIT_MESSAGE = "EAGLE is typing..."
    INPUT_PLACEHOLDER = "Type your question here..."

    # Streamlit strings
    APP_TITLE = "EAGLE"
    SESSION_STATES = "messages"

    # MongoDB strings
    USER_ROLE = "user"
    AI_ROLE = "assistant"
    ROLE_ID = "role"
    CONTENT = "content"


if __name__ == "__main__":
    llm = Ollama(
        model=os.getenv("LLM_MODEL"),
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # App title
    st.title(LLMStrings.APP_TITLE)

    # Initial prompt
    with st.chat_message(LLMStrings.AI_ROLE):
        st.write(LLMStrings.GREETINGS)

    # Initialize chat history
    if LLMStrings.SESSION_STATES not in st.session_state:
        st.session_state.messages = []

    # Connect MongoDB
    mongo_server = EasyMongo()
    collection_name = mongo_server.get_collection()

    # Display chat messages from history on app rerun
    messages = collection_name.find()
    for message in messages:
        with st.chat_message(message[LLMStrings.ROLE_ID]):
            st.markdown(message[LLMStrings.CONTENT])

    # React to user input
    if prompt := st.chat_input(LLMStrings.INPUT_PLACEHOLDER):
        # Display user message in chat message container
        with st.chat_message(LLMStrings.USER_ROLE):
            st.markdown(prompt)
            # Add user message to chat history
            user_content = create_message(LLMStrings.USER_ROLE, prompt)
            st.session_state.messages.append(user_content)

        with st.spinner(LLMStrings.WAIT_MESSAGE):
            with st.chat_message(LLMStrings.AI_ROLE):
                # Get response and display
                response = output_text(llm, prompt)

                # Add user message to chat history
                ai_content = create_message(LLMStrings.AI_ROLE, response)
                st.session_state.messages.append(ai_content)

                # Simulate stream of response with milliseconds delay
                simulate_response(response)

                # Insert messages to MongoDB
                mongo_server.insert_many([user_content, ai_content])
