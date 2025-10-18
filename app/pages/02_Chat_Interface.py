"""
Main Chat Interface Page - ChatGPT-like interface with OpenRouter LLM integration and RAG capabilities.
"""
import streamlit as st
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import custom modules
try:
    from chat.llm_client import initialize_llm_client, ConversationManager
    from chat.rag import get_rag_system
    # Import the new functions
    from chat.ui import (
        render_page_header, render_document_upload_section,
        render_chat_interface, render_chat_input, render_conversation_controls,
        render_rag_context_display, render_error_message,
        initialize_session_state, render_footer, render_chat_settings_panel
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


def main():
    """Main chat interface application."""
    try:
        # Page configuration
        st.set_page_config(
            page_title="AI Chat Interface",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Render header
        render_page_header()
        
        # Initialize systems
        llm_client = initialize_llm_client()
        rag_system = get_rag_system()
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        st.info("Please refresh the page and try again.")
        return
    
    # Initialize conversation manager
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    
    conversation_manager = st.session_state.conversation_manager
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Document upload section
        render_document_upload_section(rag_system)
        
        # Conversation controls
        st.markdown("### 🔧 Controls")
        render_conversation_controls(conversation_manager)
        
        # Chat settings panel (includes API status)
        settings = render_chat_settings_panel(llm_client)
    
    # Update conversation manager max history
    conversation_manager.max_history = settings["max_history"]
    
    with col1:
        # Check if API key is configured
        if not settings["api_key_configured"]:
            st.warning("Please configure your OpenRouter API key to start chatting")
            st.info("""
            **How to set up your API key:**
            1. Get a free API key from [OpenRouter](https://openrouter.ai/)
            2. Set environment variable: `OPENROUTER_API_KEY=your_key_here`
            3. Or add it to Streamlit secrets
            """)
            st.stop()
        
        if not llm_client:
            render_error_message("Failed to initialize LLM client")
            st.stop()
        
        # Chat interface
        render_chat_interface(conversation_manager, settings["model_name"])
        
        # Chat input
        user_input = render_chat_input()
        
        # Process user input
        if user_input:
            process_user_message(
                user_input, 
                llm_client, 
                conversation_manager, 
                rag_system, 
                settings
            )
    
    # Render footer
    render_footer()


def process_user_message(user_input: str, llm_client, conversation_manager, rag_system, settings):
    """Process user message and generate response."""
    # Add user message to conversation
    conversation_manager.add_message("user", user_input)
    
    # Check if we should use RAG
    context = ""
    rag_results = []
    
    collection_info = rag_system.get_collection_info()
    if collection_info["count"] > 0:
        # Retrieve relevant context
        context, rag_results = rag_system.retrieve_relevant_context(user_input, n_results=3)
        
        # Debug: Show if context was found
        if context.strip():
            st.info(f"Found {len(rag_results)} relevant document chunks for your question.")
        else:
            st.warning("Document is uploaded but no relevant content found for your question.")
    
    # Prepare messages for API
    system_prompt = rag_system.get_system_prompt_with_context(context)
    messages = conversation_manager.get_messages_for_api(system_prompt)
    
    # Generate response
    try:
        with st.spinner("Thinking..."):
            response = llm_client.chat_completion(
                messages=messages,
                model=settings["model_id"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"]
            )
        
        if response:
            # Add assistant response to conversation
            conversation_manager.add_message("assistant", response)
            
            # Display RAG context if used
            if context and rag_results:
                render_rag_context_display(context, rag_results)
            
            # Rerun to show new messages
            st.rerun()
        else:
            render_error_message("No response generated")
    
    except Exception as e:
        render_error_message(f"Error generating response: {str(e)}")


def render_model_info(settings):
    """Render information about the selected model."""
    if settings["model_id"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Info")
        st.sidebar.write(f"**Model:** {settings['model_name']}")
        st.sidebar.write(f"**ID:** `{settings['model_id']}`")
        
        # Model descriptions
        model_descriptions = {
            "deepseek/deepseek-r1:free": "DeepSeek R1 - Advanced reasoning model with chain-of-thought capabilities",
            "openai/gpt-oss-20b:free": "GPT OSS 20B - Open source GPT model with 20B parameters",
            "deepseek/deepseek-chat-v3-0324:free": "DeepSeek Chat V3 - Latest chat-optimized model with improved conversational abilities"
        }
        
        if settings["model_id"] in model_descriptions:
            st.sidebar.write(f"**Description:** {model_descriptions[settings['model_id']]}")


def show_usage_tips():
    """Show usage tips in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Usage Tips")
    
    with st.sidebar.expander("How to use"):
        st.markdown("""
        **Basic Chat:**
        - Just type your questions and get responses
        - Conversation history is maintained automatically
        - Adjust temperature for creativity vs consistency
        
        **Document Q&A:**
        - Upload a PDF document using the upload button
        - Ask questions about the document content
        - The AI will use document context to answer
        
        **Model Selection:**
        - Choose different models for various tasks
        - DeepSeek R1: Best for reasoning tasks
        - DeepSeek Chat V3: Best for general conversation
        - GPT OSS 20B: Alternative open source option
        """)
    
    with st.sidebar.expander("Settings"):
        st.markdown("""
        **Temperature:** Controls response randomness
        - Low (0.1-0.3): More focused, consistent
        - Medium (0.5-0.7): Balanced
        - High (0.8-1.5): More creative, varied
        
        **Max Tokens:** Response length limit
        - Shorter: Concise responses
        - Longer: More detailed responses
        
        **Max History:** Conversation memory
        - More history: Better context understanding
        - Less history: Faster responses
        """)


if __name__ == "__main__":
    main()
