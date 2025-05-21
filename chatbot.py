import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Attempt to import the tool.
# This assumes 'tools/search_tool.py' exists and 'tools' is a package
# or the script is run from the project root directory.
try:
    from tools.search_tool import execute_queries
except ImportError as e:
    print(f"Error importing 'execute_queries' from 'tools.search_tool': {e}")
    print("Please ensure 'tools/search_tool.py' exists and your PYTHONPATH is set up correctly,")
    print("or that you are running this script from the project's root directory.")
    execute_queries = None # Set to None so the script can still be loaded but will fail at agent setup

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY is used by the execute_queries tool internally

# Database schema information to help the LLM generate correct SQL queries.
# Based on tools/search_tool.py examples and comments.
DB_SCHEMA_INFO = """
The SQL database is 'data/inventory.db'.
It contains a table named 'cars' with the following columns:
  - id TEXT PRIMARY KEY: The unique identifier for a car.
  - make TEXT: The manufacturer of the car (e.g., 'Toyota', 'Honda').
  - model TEXT: The model name of the car (e.g., 'Camry', 'CRV').
  - year INTEGER: The manufacturing year of the car.
  - type TEXT: The type of car (e.g., 'Sedan', 'SUV').
  - description TEXT: A textual description of the car.

When 'use_pinecone' is true for the tool, your SQL query should try to select the 'id' column,
as these IDs can be used to filter the Pinecone semantic search.
"""

def create_chatbot_agent_executor():
    """Creates and returns the Langchain agent executor."""
    if not execute_queries:
        print("Cannot create agent executor because 'execute_queries' tool failed to import.")
        return None

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

    # The 'execute_queries' tool is already decorated with @tool and has an args_schema
    # The agent will use its name "sql-pinecone-query-executor"
    tools = [execute_queries]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a specialized assistant for answering questions about cars.
You have access to a powerful tool called "sql-pinecone-query-executor" that can query a SQL database and a Pinecone vector store.

Tool Input Schema:
The tool takes the following arguments:
1.  `sql_query` (str): An SQL query to execute against the database.
    Relevant schema information:
    {DB_SCHEMA_INFO}
2.  `pinecone_query` (str): A natural language query for Pinecone semantic search. This is used when the SQL query alone is insufficient. It can be an empty string if `use_pinecone` is false.
3.  `use_pinecone` (bool): Set to true if the user's question needs Pinecone semantic search after the SQL query. Otherwise, set to false.

Your Task:
1.  Analyze the user's question.
2.  Decide if SQL, or both are needed.
3.  Construct the appropriate `sql_query`, `pinecone_query`, and `use_pinecone` arguments for the "sql-pinecone-query-executor" tool.
4.  If `use_pinecone` is true, the `sql_query` should ideally select 'id's to help filter the Pinecone search. If no specific IDs are relevant from SQL for a Pinecone search, Pinecone will search without ID filters.
5.  If the question is very general and seems best answered by semantic search without prior SQL filtering, you can provide an empty or minimal SQL query (e.g., "SELECT id FROM cars LIMIT 0") and set `use_pinecone` to true with a relevant `pinecone_query`.
6.  If the question can be answered directly without using the tool (e.g., a simple greeting), do so.Don't use the tool for this.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor

def run_chatbot():
    """Runs the command-line interface for the chatbot."""
    print("Initializing chatbot...")
    agent_executor = create_chatbot_agent_executor()

    if not agent_executor:
        print("Chatbot initialization failed. Exiting.")
        return

    chat_history = []
    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chatbot.")
            break
        if not user_input.strip():
            continue

        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            ai_response = response.get("output", "Sorry, I couldn't process that request.")
            print(f"Chatbot: {ai_response}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=ai_response))
            # Limit chat history size to avoid overly long prompts
            if len(chat_history) > 10: # Keep last 5 pairs of messages
                chat_history = chat_history[-10:]

        except Exception as e:
            print(f"Error during agent execution: {e}")
            # Optionally, inform the user that an error occurred
            # chat_history.append(HumanMessage(content=user_input))
            # chat_history.append(AIMessage(content=f"Sorry, an error occurred while processing your request: {str(e)}"))

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please ensure it's set in your .env file or system environment.")
    elif not execute_queries:
        print("Tool 'execute_queries' could not be imported. Chatbot cannot run.")
    else:
        run_chatbot() 