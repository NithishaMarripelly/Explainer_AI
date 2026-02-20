import streamlit as st
import json
import os
import sys
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever # Using the "Classic" bridge we settled on
from langchain_core.documents import Document

# --- CONFIGURATION ---
st.set_page_config(page_title="NGO SNAP Assistant", layout="wide", page_icon="📂")
st.title("📂 SNAP Policy Assistant")
st.markdown("*Developed for NGO Resource Optimization - Powered by Hybrid RAG*")

# --- CACHED RESOURCES ---
@st.cache_resource
def load_rag_system():
    # 1. Initialize BGE-large Embeddings (Matches Notebook)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. Load Chroma DB from the path specified in your notebook
    db_path = "dbv1/chroma_db_bge"
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}. Please run your Notebook indexing first.")
        st.stop()
        
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # 3. Setup Hybrid Retriever (10 candidates for reranking)
    # Semantic Search
    vector_retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # Keyword Search (BM25) - Reconstructing from stored documents
    all_docs_data = db.get()
    docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(all_docs_data['documents'], all_docs_data['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10
    
    # Ensemble (Classic Version)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6] # Standard weighting for policy documents
    )
    
    # 4. Load Cross-Encoder Reranker (Matches Notebook Step 2)
    reranker = CrossEncoder('BAAI/bge-reranker-large')
    
    # 5. Initialize LLM (Ollama)
    llm = ChatOllama(model="llama3.1:8b", temperature=0)    
    return ensemble_retriever, reranker, llm

# Initialize the system
with st.spinner("Initializing AI Brain and Reranker..."):
    retriever, reranker_model, llm = load_rag_system()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if query := st.chat_input("Ask a SNAP policy question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and Reranking..."):
            # STEP 1: Hybrid Retrieval (10 candidates)
            initial_docs = retriever.invoke(query)
            
            # STEP 2: Reranking to Top 3 (Logic from Notebook)
            pairs = [[query, doc.page_content] for doc in initial_docs]
            scores = reranker_model.predict(pairs)
            # Combine docs with scores and sort
            scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:3]]
            
            # STEP 3: Multimodal Prompt Construction
            prompt_text = "Use the following SNAP policy context to answer. If images/tables are provided, include details from them.\n\nCONTEXT:\n"
            found_images = []
            
            for i, doc in enumerate(top_docs):
                prompt_text += f"--- Chunk {i+1} ---\n{doc.page_content}\n"
                
                # Check for tables and images in metadata (Unstructured output)
                if "tables_html" in doc.metadata:
                    prompt_text += f"TABLE DATA:\n{doc.metadata['tables_html']}\n"
                
                if "images_base64" in doc.metadata:
                    # In your notebook, this is a list of strings
                    imgs = doc.metadata["images_base64"]
                    if isinstance(imgs, list):
                        found_images.extend(imgs)
                    else:
                        found_images.append(imgs)

            prompt_text += f"\nQUESTION: {query}\nANSWER:"

            # Prepare Multimodal Message for Ollama
            message_content = [{"type": "text", "text": prompt_text}]
            
            # Add images if found (assuming Llama3-Vision or similar)
            for img_b64 in found_images[:2]: # Limit to 2 images to avoid context overflow
                message_content.append({
                    "type": "image_url",
                    "url": f"data:image/jpeg;base64,{img_b64}"
                })

            # STEP 4: Generate and Display
            try:
                msg = HumanMessage(content=message_content)
                response = llm.invoke([msg])
                full_response = response.content
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Expandable Source Viewer
                with st.expander("View Top 3 Reranked Sources"):
                    for i, (doc, score) in enumerate(scored_docs[:3]):
                        st.write(f"**Source {i+1} (Relevance Score: {score:.4f})**")
                        st.caption(doc.page_content[:400] + "...")
                        if "page_number" in doc.metadata:
                            st.write(f"📍 Document: {doc.metadata.get('filename')}, Page: {doc.metadata.get('page_number')}")
            
            except Exception as e:
                st.error(f"Error connecting to Ollama: {str(e)}")
                st.info("Make sure 'ollama serve' is running in your terminal!")