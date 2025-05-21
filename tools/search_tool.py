import sqlite3
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec # PodSpec can be imported if needed
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXPECTED_DIMENSION = 1536
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model=OPENAI_EMBEDDING_MODEL):
    """Generates an embedding for the given text using OpenAI."""
    text = text.replace("\n", " ")
    try:
        embedding_response = openai_client.embeddings.create(input=[text], model=model)
        return embedding_response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# 1. Define the Input Schema
class QueryExecutorInput(BaseModel):
    sql_query: str = Field(description="The SQL query to execute based on the database schema to answer the user's question.Always select the id information in addition to the other information from the database.")
    pinecone_query: str = Field(description="The natural language query for Pinecone semantic search. This is used when the SQL query alone is insufficient to fully answer the user's question. The semantic search runs after the SQL query is executed because the SQL query first finds the id information of the product for the semantic search. This can be an empty string if use_pinecone is false.")
    use_pinecone: bool = Field(description="A boolean flag. Set to true if the user's question cannot be completely answered using only the SQL query and database schema, requiring an additional semantic search with Pinecone. Otherwise, set to false.")

def execute_sql_query(sql_query: str) -> list[dict]:
    db_connection = sqlite3.connect("data/inventory.db")
    cursor = db_connection.cursor()
    try:
        cursor.execute(sql_query)
        sql_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        return [dict(zip(column_names, row)) for row in sql_data]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        # You might want to return an error structure or raise the exception
        # depending on how you want to handle DB errors in the tool
        return [{"error": f"SQL execution failed: {str(e)}"}]
    finally:
        if db_connection:
            db_connection.close()

def execute_pinecone_query(pinecone_query_text: str, id_list: list[str] | None = None) -> dict:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc_index = pc.Index("products")
    
    query_vector = get_embedding(pinecone_query_text)
    if not query_vector:
        return {"status": "error", "message": "Failed to generate embedding for Pinecone query."}

    query_params = {
        "vector": query_vector,
        "top_k": 10, # Or your desired top_k
        "include_metadata": True
    }

    if id_list:
        # Assuming your Pinecone metadata includes a field like 'id' or 'product_id'
        # that corresponds to the IDs from your SQL database.
        # Adjust the filter format according to your metadata structure.
        # Example: {"id": {"$in": id_list}}
        # This specific filter syntax depends on Pinecone's capabilities for filtering by a list of IDs.
        # You might need to adjust this if Pinecone expects a different filter structure for "ID in list_of_ids".
        # A common way for some vector DBs is to use a metadata field and an "$in" operator.
        # Please verify the correct filter syntax for Pinecone if this causes issues.
        query_params["filter"] = {"id": {"$in": id_list}} 
        print(f"Executing Pinecone query with text: '{pinecone_query_text}' and ID filter: {id_list}")
    else:
        print(f"Executing Pinecone query with text: '{pinecone_query_text}' without ID filter.")

    try:
        xc = pc_index.query(**query_params)
        # Assuming 'xc' is the response object from Pinecone,
        # you might want to process it further to extract just the matches or relevant data.
        return {"status": "success", "data": xc.to_dict()} # .to_dict() if it's a Pinecone response object
    except Exception as e:
        print(f"Pinecone query failed: {e}")
        return {"status": "error", "message": f"Pinecone query failed: {str(e)}"}

# 2. Define the Tool
@tool("sql-pinecone-query-executor", args_schema=QueryExecutorInput)
def execute_queries(sql_query: str, pinecone_query: str, use_pinecone: bool) -> dict:
    """
    Executes an SQL query. If 'use_pinecone' is true, it extracts 'id's from the SQL results 
    and uses them to filter a Pinecone semantic search based on 'pinecone_query'.
    The LLM should generate a full SQL query that selects 'id' if Pinecone search is anticipated.
    If the SQL query is not enough, set 'use_pinecone' to true and provide a 'pinecone_query'.
    """
    results = {}
    extracted_ids = []

    # --- SQL Query Execution ---
    print(f"Executing SQL query: {sql_query}")
    try:
        sql_query_results = execute_sql_query(sql_query)
        if sql_query_results and isinstance(sql_query_results, list) and "error" not in sql_query_results[0]:
            results["sql_results"] = {"status": "success", "data": sql_query_results}
            # Extract IDs if present - assuming the ID column is named 'id'
            # The LLM should be instructed to ensure the SQL query selects the 'id' column
            # if use_pinecone is True.
            for row in sql_query_results:
                if isinstance(row, dict) and "id" in row and row["id"] is not None:
                    extracted_ids.append(str(row["id"])) # Ensure IDs are strings if Pinecone filter expects that
            if extracted_ids:
                results["extracted_ids_for_pinecone"] = extracted_ids
                print(f"Extracted IDs for Pinecone: {extracted_ids}")
            elif use_pinecone:
                print("Warning: 'use_pinecone' is true, but no 'id's were extracted from SQL results or 'id' column was missing.")
        else:
            results["sql_results"] = {"status": "error", "data": sql_query_results} # This could be the error dict from execute_sql_query
            print(f"SQL query execution failed or returned empty/error: {sql_query_results}")
    except Exception as e:
        error_message = f"SQL execution failed: {str(e)}"
        results["sql_results"] = {"status": "error", "message": error_message}
        print(error_message)
        # If SQL fails, we might not want to proceed to Pinecone, or handle it differently
        return results

    # --- Pinecone Query Execution (Conditional) ---
    if use_pinecone:
        if not pinecone_query:
            print("Warning: 'use_pinecone' is true, but 'pinecone_query' is empty.")
            results["pinecone_results"] = {"status": "warning", "message": "Pinecone search was indicated but no query text was provided."}
        else:
            print(f"Preparing Pinecone query with text: '{pinecone_query}' and extracted IDs (if any).")
            # Pass the extracted_ids to Pinecone. If empty, it means either no IDs were found
            # or the SQL query didn't select them, or no SQL results.
            # The pinecone_query function will handle an empty or None id_list.
            pinecone_data = execute_pinecone_query(pinecone_query_text=pinecone_query, id_list=extracted_ids if extracted_ids else None)
            results["pinecone_results"] = pinecone_data
    
    return results

# --- Example Usage (for testing purposes) ---
if __name__ == '__main__':
    # Ensure PINECONE_API_KEY and OPENAI_API_KEY are set in your .env file or environment

    # Scenario 1: Only SQL query is needed (and selects id)
    print("\n--- Scenario 1: SQL only (with ID) ---")
    input_only_sql = QueryExecutorInput(
        sql_query="SELECT id, model, year FROM cars WHERE make = 'Toyota' AND year > 2020 LIMIT 2", # Assuming 'cars' table and 'id' column
        pinecone_query="",
        use_pinecone=False
    )
    result1 = execute_queries.invoke(input_only_sql.dict())
    print("Result (SQL only):")
    print(result1)

    # Scenario 2: Both SQL and Pinecone are needed
    # The SQL query should fetch IDs that will be used to filter the Pinecone search.
    print("\n--- Scenario 2: SQL and Pinecone (with ID filtering) ---")
    input_sql_and_pinecone = QueryExecutorInput(
        sql_query="SELECT id, model FROM cars WHERE type = 'SUV' AND make = 'Honda' LIMIT 3", # Fetches IDs of Honda SUVs
        pinecone_query="environmentally friendly features", # Pinecone searches for this text within the context of those Honda SUV IDs
        use_pinecone=True
    )
    result2 = execute_queries.invoke(input_sql_and_pinecone.dict())
    print("Result (SQL and Pinecone with ID filter):")
    print(result2)

    # Scenario 3: SQL query doesn't return IDs, but Pinecone is requested
    print("\n--- Scenario 3: SQL (no IDs) and Pinecone ---")
    input_sql_no_ids_pinecone = QueryExecutorInput(
        sql_query="SELECT model, year FROM cars WHERE make = 'Tesla' LIMIT 2", # No 'id' column selected
        pinecone_query="charging infrastructure",
        use_pinecone=True
    )
    result3 = execute_queries.invoke(input_sql_no_ids_pinecone.dict())
    print("Result (SQL without IDs, Pinecone without filter):")
    print(result3)
    
    # Scenario 4: SQL returns IDs, Pinecone is used with filter
    print("\n--- Scenario 4: SQL returns IDs, Pinecone uses filter ---")
    # This is similar to Scenario 2 but maybe with a different query
    # Ensure you have a data/inventory.db with a 'cars' table and 'id', 'model', 'year', 'make', 'type' columns
    # and some sample data for testing.
    # Example:
    # CREATE TABLE IF NOT EXISTS cars (id TEXT PRIMARY KEY, make TEXT, model TEXT, year INTEGER, type TEXT, description TEXT);
    # INSERT INTO cars (id, make, model, year, type, description) VALUES 
    # ('car1', 'Toyota', 'Camry', 2021, 'Sedan', 'Reliable family sedan with good fuel economy.'),
    # ('car2', 'Honda', 'CRV', 2022, 'SUV', 'Popular compact SUV with ample cargo space.'),
    # ('car3', 'Toyota', 'Rav4', 2020, 'SUV', 'Versatile SUV, great for city and adventure.'),
    # ('car4', 'Honda', 'Civic', 2023, 'Sedan', 'Sporty and efficient compact car.'),
    # ('car5', 'Honda', 'Pilot', 2022, 'SUV', 'Large SUV for families, lots of features.');
    
    # For Pinecone, you'd need to have vectors in your "products" index.
    # Those vectors should have metadata like {"id": "car1", "text_chunk": "..."}
    # where "text_chunk" is what you're semantically searching.
    # The execute_pinecone_query is set up to filter by these IDs.
    # If your actual IDs are integers in SQL, make sure they are stored/filtered as strings or integers consistently.

    # Test with SQL that should return some IDs (e.g., from your actual DB)
    # For example, if you have products with IDs 'product_A', 'product_B'
    test_sql_with_ids = QueryExecutorInput(
        sql_query="SELECT id, model FROM cars WHERE make = 'Honda'", # Assuming 'cars' table in inventory.db
        pinecone_query="safety features",
        use_pinecone=True
    )
    # print("\n--- Scenario 4: Testing with actual DB query for IDs ---")
    # result4 = execute_queries.invoke(test_sql_with_ids.dict())
    # print("Result (Scenario 4):")
    # print(result4)
