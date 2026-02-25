import streamlit as st
import os
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Page config
st.set_page_config(
    page_title="RAG Policy Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'db' not in st.session_state:
    st.session_state.db = None
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input (will use secrets in production)
    api_key = os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get free API key from console.groq.com"
        )
    
    # Model selection
    model_options = {
        "Qwen 2.5 7B (Fast)": "qwen2.5-7b-instruct",
        "Qwen 2.5 14B (Better)": "qwen2.5-14b-instruct",
        "Qwen 2.5 32B (Best Balance)": "qwen2.5-32b-instruct",
        "Qwen 2.5 72B (Highest Quality)": "qwen2.5-72b-instruct",
    }
    
    selected_model_name = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    
    model_name = model_options[selected_model_name]
    
    # Initialize Groq client
    if api_key and st.button("🚀 Initialize"):
        try:
            st.session_state.groq_client = Groq(api_key=api_key)
            
            # Test the connection
            test_response = st.session_state.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model=model_name,
                max_tokens=10
            )
            
            st.success("✅ Connected to Groq successfully!")
            st.info(f"Using model: {selected_model_name}")
            
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.divider()
    
    # Vector DB settings
    st.subheader("Vector Database")
    
    db_path = st.text_input(
        "ChromaDB Path",
        value="./chroma_db",
        help="Path to your ChromaDB vector store"
    )
    
    if st.button("🔄 Load Vector DB"):
        with st.spinner("Loading vector database..."):
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.db = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings
                )
                st.success("✅ Vector DB loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load DB: {str(e)}")
    
    st.divider()
    
    # Generation settings
    st.subheader("Generation Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    top_k = st.slider("Top K Results", 1, 10, 3)
    max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100)
    
    st.divider()
    
    # Status
    st.subheader("System Status")
    st.write("🤖 LLM:", "✅ Connected" if st.session_state.groq_client else "⚠️ Not initialized")
    st.write("💾 Vector DB:", "✅ Loaded" if st.session_state.db else "⚠️ Not loaded")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("📚 RAG Policy Assistant")
st.markdown("Ask questions about your policy documents using cloud-hosted AI")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(source[:300] + "..." if len(source) > 300 else source)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your policies..."):
    # Check if system is ready
    if not st.session_state.groq_client:
        st.error("⚠️ Please initialize the Groq client first!")
        st.stop()
    
    if not st.session_state.db:
        st.error("⚠️ Please load the vector database first!")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant documents
                retriever = st.session_state.db.as_retriever(
                    search_kwargs={"k": top_k}
                )
                relevant_docs = retriever.invoke(prompt)
                
                # Create context
                context = "\n\n".join([
                    f"Document {i+1}:\n{doc.page_content}" 
                    for i, doc in enumerate(relevant_docs)
                ])
                
                # Generate answer using Groq
                rag_prompt = f"""You are a helpful policy assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {prompt}

Answer based on the context above. If the answer is not in the context, say so."""

                response = st.session_state.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful policy assistant."},
                        {"role": "user", "content": rag_prompt}
                    ],
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                answer = response.choices[0].message.content
                
                # Display answer
                st.markdown(answer)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": [doc.page_content for doc in relevant_docs]
                })
                
                # Show sources
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.divider()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Powered by Groq Cloud • Free API • No PC Required
</div>
""", unsafe_allow_html=True)