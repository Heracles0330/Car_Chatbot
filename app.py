import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Attempt to import the agent executor creation function from chatbot.py
try:
    from chatbot import create_chatbot_agent_executor, OPENAI_API_KEY, execute_queries
except ImportError as e:
    st.error(f"Failed to import necessary components from chatbot.py: {e}")
    st.error("Please ensure chatbot.py is in the same directory and all its dependencies are met.")
    OPENAI_API_KEY = None
    execute_queries = None
    create_chatbot_agent_executor = None

# Page Configuration: Set to wide layout first
st.set_page_config(layout="wide", page_title="RC Parts Pro Chatbot")

# --- Sidebar Configuration ---
with st.sidebar:
    # Replace "assets/rc_logo.png" with the actual path to your company's logo
    # For example, if your logo is in an 'assets' folder: "assets/logo.png"
    try:
        st.image("assets/rcsuperstore_logo.webp", width=300)  # Adjust width as needed
    except FileNotFoundError:
        st.warning("Company logo not found in sidebar. Expected at 'assets/rc_logo.png'.")
    except Exception as e:
        st.warning(f"Could not load logo: {e}")

    st.title("RC Parts Pro 🚗💨") # Sidebar title
    st.markdown(
        "Welcome! I'm your AI assistant for finding RC car parts and products. "
        "Ask me about specific items, compatibility, or features."
    )
    st.divider()
    st.image("assets/car.jpg", width=300)

# --- Main Chat Interface ---
st.title("Chat with RC Parts Pro") # Main panel title
st.divider()

# Initialize or get the agent executor from session state
if "agent_executor" not in st.session_state:
    if create_chatbot_agent_executor and OPENAI_API_KEY and execute_queries:
        with st.spinner("🤖 Initializing Chatbot Agent... Please wait."):
            st.session_state.agent_executor = create_chatbot_agent_executor()
        if st.session_state.agent_executor is None:
            st.error("Chatbot agent failed to initialize. Check API keys and configurations.")
    else:
        st.session_state.agent_executor = None
        # Warnings for missing components are good for initial setup
        if not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY not found. Please ensure it's set in your .env file.")
        if not execute_queries:
            st.warning("The 'execute_queries' tool (from chatbot.py) is not available. The chatbot may not function.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content, unsafe_allow_html=True)

# Get user input
user_query = st.chat_input("Ask your question about RC parts or products...")

if user_query:
    st.session_state.current_ai_full_response_for_history = ""
    if not st.session_state.agent_executor:
        st.error("Chatbot is not initialized. Cannot process query.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            with st.spinner("🔍 Thinking & Searching..."):
                
                stream = st.session_state.agent_executor.stream({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history[:-1]
                })
                for chunk in stream:
                    if "output" in chunk and isinstance(chunk["output"], str) and chunk["output"]:
                        st.write(chunk["output"])
                        st.session_state.current_ai_full_response_for_history += chunk["output"]
                    
            
        
        st.session_state.chat_history.append(AIMessage(content=st.session_state.current_ai_full_response_for_history))

        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]

if not st.session_state.agent_executor and (OPENAI_API_KEY and execute_queries):
    st.error("Chatbot initialization failed. Please check terminal output or error messages for details.")
