# AI Chat Interface with RAG

A ChatGPT-like interface built with Streamlit that uses OpenRouter API to access free open-source LLMs with document Q&A capabilities.

## Features

**Multiple LLM Models**
- GPT OSS 20B (Default)
- DeepSeek R1
- DeepSeek Chat V3
- Qwen 3 30B
- Dolphin Mistral 24B (uncensored)

**Chat Features**
- Conversation memory (maintains context)
- Adjustable response creativity (temperature)
- Token limit control
- Export chat history

**Document Q&A (RAG)**
- Upload PDF documents (automatic processing)
- Ask questions about document content
- Vector search with context retrieval
- Document management with clear all option

## Setup

### 1. Get OpenRouter API Key
1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Create account and get your free API key
3. The models we use are completely free (Message Limit: 50/day)!

### 2. Configure API Key

**Option A: Environment Variable**
```bash
set OPENROUTER_API_KEY=your_key_here
```

**Option B: .env File**
```bash
# Copy .env.example to .env and fill in your key
cp .env.example .env
# Edit .env and add your key
```

**Option C: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "your_key_here"
```

## Usage

### Basic Chat
1. Check API key status in the right panel
2. Select your preferred model from the right panel settings
3. Adjust temperature and token settings as needed
4. Type your message and press Enter
5. The AI will respond with context from previous messages

### Document Q&A
1. Upload a PDF document using the file uploader (right panel)
2. Document is processed automatically (no button click needed)
3. Ask questions about the document content
4. The AI will use relevant document sections to answer
5. Use "Clear All" to remove documents when done

### Model Selection Guide

- **GPT OSS 20B**: Default choice, good general-purpose model
- **DeepSeek R1**: Best for complex reasoning, math, analysis
- **DeepSeek Chat V3**: Best for general conversation, creative tasks
- **Qwen 3 30B**: Excellent for multilingual tasks and complex queries
- **Dolphin Mistral 24B**: Creative writing and uncensored responses

### Settings (Right Panel)

- **API Status**: Shows if your OpenRouter key is configured
- **Model Selection**: Choose from 5 free models
- **Temperature**: Controls creativity (0.0 = focused, 1.5 = creative)
- **Max Tokens**: Response length limit (100-4000)
- **Max History**: How many messages to remember (0-50)

## Architecture

### Components

- `llm_client.py`: OpenRouter API integration with conversation management
- `rag.py`: Document processing, embedding, and vector search
- `ui.py`: Reusable Streamlit UI components
- `04_Chat_Interface.py`: Main chat page

### Data Flow

1. User input → Conversation Manager
2. RAG system retrieves relevant document context (if available)
3. LLM Client sends request to OpenRouter with full context
4. Response displayed and added to conversation history

### Vector Storage

- Uses ChromaDB for document embedding storage
- Sentence Transformers for text embeddings
- Persistent storage between sessions
- Semantic search for context retrieval

## Interface Layout

### Left Panel (Chat Area)
- **Chat History**: Conversation with clear user/assistant message styling
- **Chat Input**: Type messages at the bottom
- **Model Name**: Current model displayed in chat header
- **Context Display**: RAG context shown when documents are used

### Right Panel (Controls)
- **Documents**: Upload PDFs with automatic processing
- **Controls**: New Chat and Export Chat buttons  
- **Chat Settings**: API status, model selection, and parameters

### Clean Design
- Grey background for user messages
- Light green background for assistant messages
- No emojis in assistant responses for clean appearance
- Responsive layout that works on different screen sizes

## Models Information

All models are accessed via OpenRouter's free tier:

- **GPT OSS 20B** (`openai/gpt-oss-20b:free`) - **DEFAULT**
  - Open source model
  - 20 billion parameters
  - Good general-purpose performance

- **DeepSeek R1** (`deepseek/deepseek-r1:free`)
  - Advanced reasoning capabilities
  - Chain-of-thought processing
  - Great for complex analysis

- **DeepSeek Chat V3** (`deepseek/deepseek-chat-v3-0324:free`)
  - Optimized for conversation
  - Balanced performance
  - Good general-purpose model

- **Qwen 3 30B** (`qwen/qwen3-30b-a3b:free`)
  - Large 30 billion parameter model
  - Excellent multilingual capabilities
  - Strong performance on complex tasks

- **Dolphin Mistral 24B** (`cognitivecomputations/dolphin-mistral-24b-venice-edition:free`)
  - Creative and uncensored responses
  - Good for creative writing
  - 24 billion parameters

## Troubleshooting

### API Key Issues
- Check environment variables with `echo $OPENROUTER_API_KEY`
- Verify key is valid at OpenRouter.ai
- Make sure no extra spaces in key

### Import Errors
- Install all dependencies: `pip install -r requirements.txt`
- Check Python environment is activated
- Try reinstalling problematic packages

### Document Processing Issues
- Only PDF files are supported currently
- Large files may take time to process automatically
- Check file isn't corrupted or password-protected
- Files are processed immediately upon upload (no button needed)
- Use "Clear All" to remove documents and reset processing state

### Model Responses
- If responses are cut off, increase max tokens
- For better responses, adjust temperature
- Clear chat history if context becomes confusing
