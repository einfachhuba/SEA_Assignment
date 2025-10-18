"""
Streamlit UI components for the chat interface.
"""
import streamlit as st
from typing import Dict, List, Optional, Any
import os


def render_sidebar_api_status() -> Dict[str, Any]:
    """Render sidebar with only API key status."""
    with st.sidebar:
        st.header("⚙️ API Status")
        
        # API Key status
        api_key_status = check_api_key_status()
        if api_key_status["has_key"]:
            st.success("OpenRouter API key configured")
        else:
            st.error("OpenRouter API key not found")
            st.info("Add OPENROUTER_API_KEY to your environment variables or Streamlit secrets")
            if st.button("Get API Key"):
                st.link_button("Get OpenRouter API Key", "https://openrouter.ai/")
        
        return api_key_status


def render_chat_settings_panel(llm_client) -> Dict[str, Any]:
    """Render chat settings panel for right side."""
    st.subheader("⚙️ Chat Settings")
    
    # API Key status
    api_key_status = check_api_key_status()
    if api_key_status["has_key"]:
        st.success("OpenRouter API key configured")
    else:
        st.error("OpenRouter API key not found")
        st.info("Add OPENROUTER_API_KEY to your environment variables or Streamlit secrets")
        if st.button("Get API Key", key="get_api_key_btn"):
            st.link_button("Get OpenRouter API Key", "https://openrouter.ai/")
    
    # Model selection
    st.write("**Model Selection**")
    if llm_client:
        available_models = llm_client.get_available_models()
        model_names = list(available_models.keys())
        
        # Default to GPT OSS 20B
        default_model = "GPT OSS 20B" if "GPT OSS 20B" in model_names else model_names[0]
        default_index = model_names.index(default_model) if default_model in model_names else 0
        
        selected_model_name = st.selectbox(
            "Choose Model:",
            model_names,
            index=default_index,
            help="Select the LLM model to use for chat",
            key="model_selector"
        )
        
        selected_model_id = available_models[selected_model_name]
    else:
        selected_model_name = "No models available"
        selected_model_id = None
        st.error("LLM client not initialized")
    
    # Chat parameters
    st.write("**Parameters**")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses. Lower = more focused, Higher = more creative",
            key="temp_slider"
        )
    
    with col2:
        max_history = st.slider(
            "Max History",
            min_value=0,
            max_value=50,
            value=20,
            help="Maximum number of messages to keep in conversation history",
            key="history_slider"
        )
    
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=1000,
        step=100,
        help="Maximum number of tokens in the response",
        key="tokens_input"
    )
    
    return {
        "model_name": selected_model_name,
        "model_id": selected_model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_history": max_history,
        "api_key_configured": api_key_status["has_key"]
    }


def render_sidebar_settings(llm_client) -> Dict[str, Any]:
    """Render sidebar with model selection and settings."""
    with st.sidebar:
        st.header("⚙️ Chat Settings")
        
        # API Key status
        api_key_status = check_api_key_status()
        if api_key_status["has_key"]:
            st.success("OpenRouter API key configured")
        else:
            st.error("OpenRouter API key not found")
            st.info("Add OPENROUTER_API_KEY to your environment variables or Streamlit secrets")
            if st.button("Get API Key"):
                st.link_button("Get OpenRouter API Key", "https://openrouter.ai/")
        
        # Model selection
        st.subheader("Model Selection")
        if llm_client:
            available_models = llm_client.get_available_models()
            model_names = list(available_models.keys())
            
            # Default to GPT OSS 20B
            default_model = "GPT OSS 20B" if "GPT OSS 20B" in model_names else model_names[0]
            default_index = model_names.index(default_model) if default_model in model_names else 0
            
            selected_model_name = st.selectbox(
                "Choose Model:",
                model_names,
                index=default_index,
                help="Select the LLM model to use for chat"
            )
            
            selected_model_id = available_models[selected_model_name]
        else:
            selected_model_name = "No models available"
            selected_model_id = None
            st.error("LLM client not initialized")
        
        # Chat parameters
        st.subheader("Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses. Lower = more focused, Higher = more creative"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum number of tokens in the response"
        )
        
        # Conversation settings
        st.subheader("Conversation")
        max_history = st.slider(
            "Max History Messages",
            min_value=0,
            max_value=50,
            value=20,
            help="Maximum number of messages to keep in conversation history"
        )
        
        return {
            "model_name": selected_model_name,
            "model_id": selected_model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_history": max_history,
            "api_key_configured": api_key_status["has_key"]
        }


def render_document_upload_section(rag_system) -> bool:
    """Render document upload and management section."""
    st.subheader("📄 Document Upload (RAG)")
    
    # Collection info
    collection_info = rag_system.get_collection_info()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if collection_info["count"] > 0:
            st.info(f"{collection_info['count']} document chunks stored")
            if collection_info["documents"]:
                st.write("**Documents:**", ", ".join(collection_info["documents"]))
    
    with col2:
        if collection_info["count"] > 0:
            if st.button("Clear All", help="Remove all uploaded documents"):
                rag_system.clear_documents()
                # Clear processed file tracking
                for key in list(st.session_state.keys()):
                    if key.startswith("processed_"):
                        del st.session_state[key]
                st.rerun()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF document",
        type=['pdf'],
        help="Upload a PDF document - it will be processed automatically",
        key="pdf_uploader"
    )
    
    # Auto-process when file is uploaded
    if uploaded_file is not None:
        # Check if this file has already been processed
        if f"processed_{uploaded_file.name}" not in st.session_state:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                success = rag_system.process_and_store_document(uploaded_file)
                if success:
                    st.session_state[f"processed_{uploaded_file.name}"] = True
                    st.success(f"{uploaded_file.name} processed successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to process {uploaded_file.name}")
                return success
        else:
            st.success(f"{uploaded_file.name} is already processed")
    
    return False


def render_chat_interface(conversation_manager, selected_model_name: str = None) -> None:
    """Render the main chat interface."""
    if selected_model_name:
        st.subheader(f"💬 Chat - {selected_model_name}")
    else:
        st.subheader("💬 Chat")
    
    # Display conversation history
    conversation_history = conversation_manager.get_conversation_history()
    
    if not conversation_history:
        st.info("Start a conversation! You can ask general questions or upload a document to ask questions about it.")
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for msg in conversation_history:
            if msg["role"] == "user":
                # User message - grey with dark text
                st.markdown(
                    f'<div style="background-color: #A5A5A5FF; color: #111111; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #666666;">'
                    f'<strong style="color: #111111;">You:</strong> {msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                # Assistant message - light green with dark text
                st.markdown(
                    f'<div style="background-color: #C2C2C2FF; color: #111111; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #4a9a4a;">'
                    f'<strong style="color: #111111;">Assistant:</strong> {msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )


def render_chat_input() -> Optional[str]:
    """Render chat input and return user message."""
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        return user_input.strip()
    
    return None


def render_conversation_controls(conversation_manager) -> None:
    """Render conversation control buttons."""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("New Chat"):
            conversation_manager.clear_history()
            st.rerun()
    
    with col2:
        history = conversation_manager.get_conversation_history()
        if history:
            if st.button("Export Chat"):
                export_conversation(history)


def export_conversation(conversation_history: List[Dict[str, str]]) -> None:
    """Export conversation history as downloadable text."""
    if not conversation_history:
        st.warning("No conversation to export")
        return
    
    # Format conversation for export
    export_text = "# Chat Conversation Export\n\n"
    for msg in conversation_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        export_text += f"**{role}:** {msg['content']}\n\n"
    
    st.download_button(
        label="Download Chat",
        data=export_text,
        file_name="chat_conversation.txt",
        mime="text/plain"
    )


def check_api_key_status() -> Dict[str, Any]:
    """Check if OpenRouter API key is configured."""
    api_key = None
    
    # Check environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Check Streamlit secrets
    if not api_key and hasattr(st, 'secrets'):
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY")
        except:
            pass
    
    return {
        "has_key": bool(api_key and api_key.strip()),
        "source": "environment" if os.getenv("OPENROUTER_API_KEY") else "secrets" if api_key else None
    }


def render_rag_context_display(context: str, results: List[Dict]) -> None:
    """Display RAG context and sources in an expander."""
    if context and results:
        with st.expander("Document Context Used", expanded=False):
            st.write("**Retrieved context for this response:**")
            
            for i, result in enumerate(results):
                with st.container():
                    st.write(f"**Chunk {i+1}:**")
                    st.write(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])
                    
                    # Show metadata if available
                    if 'metadata' in result and result['metadata']:
                        metadata = result['metadata']
                        if 'document_name' in metadata:
                            st.caption(f"📄 Source: {metadata['document_name']}")
                    
                    if i < len(results) - 1:
                        st.divider()


def render_error_message(error_message: str) -> None:
    """Render error message in a consistent format."""
    st.error(f"{error_message}")


def render_success_message(success_message: str) -> None:
    """Render success message in a consistent format."""
    st.success(f"{success_message}")


def render_info_message(info_message: str) -> None:
    """Render info message in a consistent format."""
    st.info(f"{info_message}")


def initialize_session_state() -> None:
    """Initialize session state variables for the chat interface."""
    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = True
    
    if "show_rag_context" not in st.session_state:
        st.session_state.show_rag_context = False


def render_page_header() -> None:
    """Render the main page header."""
    st.title("AI Chat Interface")
    st.markdown("Chat with AI models via OpenRouter API with document Q&A capabilities")


def render_footer() -> None:
    """Render page footer with information."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>
                Powered by OpenRouter API -  
                <a href='https://openrouter.ai/' target='_blank'>Get your API key</a>
            </small>
        </div>
        """, 
        unsafe_allow_html=True
    )
