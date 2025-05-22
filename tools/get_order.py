import sqlite3
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec # PodSpec can be imported if needed
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
load_dotenv()


# 1. Define the Input Schema
class GetOrderInput(BaseModel):
    orderId: str = Field(description="This order id is for the order in BigCommerce API to get the order details.")
    

# 2. Define the Tool
@tool("get-order", args_schema=GetOrderInput)
def get_order(orderId: str) -> dict:
    """
    This tool is used to get the order details from BigCommerce API. This tool uses the orderId to get the order details.
    This tool uses  https://api.bigcommerce.com/stores/{store_hash}/v2/orders/{order_id}.
    """
    store_hash = os.getenv("BIGCOMMERCE_STORE_HASH")
    headers = {
        "X-Auth-Token": os.getenv("BIGCOMMERCE_API_KEY"),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.get(f"https://api.bigcommerce.com/stores/{store_hash}/v2/orders/{orderId}", headers=headers)
    if response.status_code != 200:
        return {"error": f"Failed to fetch order. Status code: {response.status_code}", "details": response.text}
    data = response.json()
    return data

if __name__ == "__main__":
    print(get_order("171"))