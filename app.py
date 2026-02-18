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
    
    # Load the existing database
    # Make sure this path matches your actual folder structure
    db = Chroma(persist_directory="dbv1/chroma_db", embedding_function=embeddings)
    
    # Initialize the LLM
    # Using the model and temp you specified
    llm = ChatOllama(model="qwen2.5:14b", temperature=0.2)
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
    # Append user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- RAG LOGIC ---
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            
            # 1. Retrieve Context
            retriever = db.as_retriever(search_kwargs={"k": 3})
            chunks = retriever.invoke(prompt)
            
            # 2. Build the Strict System Prompt
            # This follows your requested format strictly
            prompt_text = (
                "You are a strict Knowledge Base Assistant. Your sole purpose is to answer the user's "
                "question based ONLY on the provided documents.\n\n"
                "*** INSTRUCTIONS ***\n"
                "1. **Analyze First:** Before answering, think step-by-step. Scan the documents to identify "
                "exact sentences or data points that support the answer.\n"
                "2. **Evidence Check:** If you cannot find supporting sentences in the documents, "
                "you MUST state 'The answer is not available in the provided documents.'\n"
                "3. **Format:** You must structure your response exactly as follows:\n"
                "   --- REASONING ---\n"
                "   (List the key facts or quotes you found that support the answer)\n"
                "   --- FINAL ANSWER ---\n"
                "   (The direct answer to the user's question)\n\n"
                "*** END INSTRUCTIONS ***\n\n"
                f"USER QUESTION: {prompt}\n\n"
                "CONTENT TO ANALYZE:\n"
            )

            # 3. Initialize Message Content with Text
            message_content = []
            
            # Loop through chunks to add Text and Tables to prompt_text
            # and collect Images for the message payload
            found_images = []

            for i, chunk in enumerate(chunks):
                prompt_text += f"--- Document {i+1} ---\n"
                
                if "original_content" in chunk.metadata:
                    original_data = json.loads(chunk.metadata["original_content"])
                    
                    # Add Text
                    raw_text = original_data.get("raw_text", "")
                    if raw_text:
                        prompt_text += f"TEXT:\n{raw_text}\n\n"
                    
                    # Add Tables
                    tables_html = original_data.get("tables_html", [])
                    if tables_html:
                        prompt_text += "TABLES:\n"
                        for j, table in enumerate(tables_html):
                            prompt_text += f"Table {j+1}:\n{table}\n\n"
                            
                    # Collect Images (Don't add to prompt_text, add to payload later)
                    images_base64 = original_data.get("images_base64", [])
                    for img_b64 in images_base64:
                        found_images.append(img_b64)

            prompt_text += "\nANSWER:"

            # Add the final constructed text prompt to the message
            message_content.append({"type": "text", "text": prompt_text})

            # Add all found images to the message payload
            for img_b64 in found_images:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            # 4. Generate Answer
            try:
                msg = HumanMessage(content=message_content)
                response = llm.invoke([msg])
                full_response = response.content
                
                # Display the response
                st.markdown(full_response)
                
                # Append assistant response to state
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Optional: Show Sources in an expander for debugging/transparency
                with st.expander("View Source Chunks"):
                    for i, chunk in enumerate(chunks):
                        st.write(f"**Source {i+1}:**")
                        st.caption(chunk.page_content[:300] + "...")

            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})