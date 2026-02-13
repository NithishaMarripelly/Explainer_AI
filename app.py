import streamlit as st
import json
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# --- CONFIGURATION ---
st.set_page_config(page_title="NGO Food Equity Monitor", layout="wide")
st.title("📂 SNAP Policy Assistant")
st.markdown("Developed for NGO Resource Optimization")

# --- CACHED RESOURCES ---
@st.cache_resource
def load_rag_system():
    # Load the same embeddings you used for ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the existing database (dbv1 or dbv2 folder)
    db = Chroma(persist_directory="dbv1/chroma_db", embedding_function=embeddings)
    
    # Initialize the LLM
    llm = ChatOllama(model="llama3.2", temperature=0)
    return db, llm

db, llm = load_rag_system()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a policy question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- RAG LOGIC ---
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            # 1. Retrieve
            retriever = db.as_retriever(search_kwargs={"k": 3})
            chunks = retriever.invoke(prompt)
            
            # 2. Build Multimodal Prompt (Your existing logic)
            prompt_text = f"Based on the documents, answer: {prompt}\n\nCONTEXT:\n"
            message_content = [{"type": "text", "text": prompt_text}]
            
            for chunk in chunks:
                if "original_content" in chunk.metadata:
                    data = json.loads(chunk.metadata["original_content"])
                    prompt_text += f"\nTEXT: {data.get('raw_text', '')}\n"
                    # Add images to message_content if available
                    for img_b64 in data.get("images_base64", []):
                        message_content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        })
            
            # 3. Generate Answer
            msg = HumanMessage(content=message_content)
            response = llm.invoke([msg])
            full_response = response.content
            
            st.markdown(full_response)
            
            # Optional: Show Sources in an expander
            with st.expander("View Source Chunks"):
                for i, chunk in enumerate(chunks):
                    st.write(f"**Source {i+1}:** {chunk.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": full_response})