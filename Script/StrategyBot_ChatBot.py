import streamlit as st
import os
import csv
from datetime import datetime
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole



# ----- Sidebar -----
with st.sidebar:
    st.image("[insert_file_path]/giuseppe-buccola-zcKLLpMAbXU-unsplash.jpg", use_container_width=True) #insert actual file path for image
    st.caption("Photo by [Giuseppe Buccola](https://unsplash.com/photos/zcKLLpMAbXU) on Unsplash")
    st.title("Hello! I'm StrategyBot")
    st.markdown("Here to help you in your journey with developing a strategy.")
    st.markdown("""
        <style>
            .stButton button {
                background-color: #0066cc;  /* Blue color */
                color: white;  /* Text color */
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
            }
            .stButton button:hover {
                background-color: #0056b3;  /* Darker blue when hovered */
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Refresh button functionality    
    if st.button("🔄 Want to restart the conversation?"):
        st.session_state["chat_history"] = []
        
        st.success("Conversation history has been cleared. We can start fresh 😊")
        
# Download buttons for source PDFs
    st.markdown("---")
    st.markdown("### Download data sources:")
    
    st.download_button(
        label="Complete Guide to Strategic Planning",
        data=open("[insert_file_path]/Complete-Guide-to-Strategic-Planning.pdf", "rb").read(), #insert actual file path
        file_name="Complete-Guide-to-Strategic-Planning.pdf",
        mime="application/pdf")
    
    st.download_button(
        label="How to formulate a successful business strategy",
        data=open("[insert_file_path]/how-to-formulate-successful-business-strategy.pdf", "rb").read(), #insert actual file path
        file_name="how-to-formulate-successful-business-strategy.pdf",
        mime="application/pdf")



# ----- Search Engine Setup -----
documents = SimpleDirectoryReader("[insert_file_path]/Data").load_data()#create Data folder with PDFs and include it's path here

# Configuration for HuggingFace model
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(
    model_name=hf_model,
    task="text-generation",
    token="[insert_HF_API_token]" #sign up for HuggingFace and get API token; insert here
)


# Embedding model
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder="[insert_file_path]/Embeddings"  #create Embeddings folder and include it's path here
)

#---------------Vector Index Creation (First run only to generate and save vector DB)
text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)

# Uncomment for the first run
vector_index = VectorStoreIndex.from_documents(
     documents,
     transformations=[text_splitter],
     embed_model=embeddings)

vector_index.storage_context.persist(persist_dir="[insert_file_path]/Vector_index") #create Vector_index folder and include it's path here

#---------------For continued runs -----------
# Uncomment next lines after first run above to access generated and saved vector DB (need to comment out first-run section)

#storage_context = StorageContext.from_defaults(persist_dir="[insert_file_path]/Vector_index_single_file") #create Vector_index folder and include it's path here
#vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

# ---------------ChatBot Setup -----------

retriever = vector_index.as_retriever(similarity_top_k=2)

memory = ChatMemoryBuffer.from_defaults()


prompts = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are StrategyBot, a helpful, friendly, and concise assistant. "
            "Only answer the user's question using the provided context documents and the current conversation. "
            "Get straight to the point and reply in a few words (max 50 characters). "
            "If the answer is not in the context, reply with: 'I don’t know based on the provided information'. "
            "Use clear and simple language."
        )
    )
]





@st.cache_resource
def init_bot():
    return ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prompts)

rag_bot = init_bot()

# ----- Page presentation -----
st.title("Strategy Development: Guidance at Your Fingertips")

st.markdown("""
Ask me anything about the process of developing, implementing, and monitoring a business strategy.  
My answers are based on **On Strategy's** [*Complete Guide to Strategic Planning*](https://onstrategyhq.com/complete-strategy-guide/) and **Harvard Business School's** [*How to Formulate a Successful Business Strategy*](https://info.email.online.hbs.edu/strategy-formulation-ebook)
""")

# ----- Example questions to guide the user -----
st.markdown(" **Sample Questions:**")
st.markdown("- What are the steps in developing a business strategy?")
st.markdown("- How can I measure the success of a strategy?")
st.markdown("- What should be included in the implementation phase?")


# ----- Display chat history -----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history: 
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    
if user_input := st.chat_input("What can I help with today?"):
    st.chat_message("human", avatar="[insert_file_path]/magnifying-glass.png").markdown(user_input) #insert actual file path for image
    st.session_state.chat_history.append({"role": "human", "content": user_input})

    with st.spinner("📂 Searching strategy documents..."):
        try:
            answer = rag_bot.chat(user_input).response

        except Exception as e:
            answer = f"Sorry, I had trouble processing your question: {e}"

    with st.chat_message("assistant", avatar= "[insert_file_path]/bot.png"): #insert actual file path for image
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

        
# ----- Function to save feedback -----
def log_feedback(feedback_data):
    folder_path = "[insert_file_path]/ChatBot_Feedback" #create ChatBot_Feedback folder and include it's path here
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "chat_feedback_log.csv")
    
    file_exists = os.path.exists(file_path)
    
    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "question", "answer", "feedback"])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(feedback_data)
  

# ----- Collect Question, Answer, and Feedback -----
if st.session_state.chat_history:
    if len(st.session_state.chat_history) > 1:
        last_message = st.session_state.chat_history[-1]
        if last_message["role"] == "assistant":
            feedback_key = f"feedback_radio_{len(st.session_state.chat_history)}"
            
            # Display feedback prompt
            st.markdown("### Was this answer helpful?")
            feedback = st.radio(" ", ("👍 Yes", "👎 No"), index=None, key=feedback_key)
            
            # When feedback is submitted
            if feedback:
                st.write("Thank you for your feedback!")
                
                # Collect question, answer, and feedback
                user_question = st.session_state.chat_history[-2]["content"]  # User's question
                assistant_answer = last_message["content"]  # Assistant's answer
                
                # Prepare the data to log
                feedback_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": user_question,
                    "answer": assistant_answer,
                    "feedback": feedback,
                }

                # Save the feedback to a file (continuously appending)
                log_feedback(feedback_data)
                
                # Reset the radio button for next question/answer pair
                del st.session_state[feedback_key]  


# ----- Performance Optimization -----
@st.cache_resource
def load_documents():
    return SimpleDirectoryReader("[insert_file_path]/Data").load_data() #include file path to Data folder with PDFs

documents = load_documents()  # Caching documents to improve load time
