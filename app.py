import os
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline

# 1. PAGE SETUP
st.set_page_config(page_title="Policy Navigator", page_icon="📄")
st.title("📄 Medicaid Policy Assistant")

# 2. RESOURCE LOADING (Cached)
@st.cache_resource
def load_rag_assets():
    # Use the EXACT same embedding model from your notebook
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Rebuild storage context from the local directory
    if not os.path.exists("./storage"):
        st.error("Storage folder not found! Please run the notebook build first.")
        st.stop()
        
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    
    # Load index and EXPLICITLY pass the embed_model
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    
    # Initialize the local LLM for generation
    # FLAN-T5 base is recommended for meaningful summaries
    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    
    return index, qa_pipe, embed_model

index, qa_pipe, embed_model = load_rag_assets()

# 3. INTERACTIVE UI
query = st.text_input("Ask a policy question:")

if st.button("Get Answer") and query:
    with st.spinner("Analyzing documents..."):
        # Retrieve context (Top 3 relevant chunks)
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(query)
        
        # Combine retrieved content into one block
        context = "\n\n".join([n.get_content() for n in nodes])
        
        # Construct an "Elaborative" Prompt
        prompt = (
            f"Context: {context}\n\n"
            f"Instruction: Based on the context above, provide a detailed and "
            f"meaningful answer to the question: {query}\n"
            f"Answer:"
        )
        
        # Generate with length constraints to prevent one-word answers
        answer = qa_pipe(
            prompt, 
            max_length=300, 
            min_length=20, 
            repetition_penalty=2.5
        )[0]['generated_text']
        
        st.subheader("Answer")
        st.write(answer)
        
        with st.expander("View Source Documents"):
            for i, node in enumerate(nodes):
                st.info(f"Source {i+1}:\n{node.get_content()[:500]}...")