import sqlite3
import openai
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to the database
conn = sqlite3.connect("data/inventory.db")
cursor = conn.cursor()

# 1. Add the new column (run only once)
try:
    cursor.execute("ALTER TABLE products ADD COLUMN global_type TEXT;")
    conn.commit()
except sqlite3.OperationalError:
    print("Column already exists, skipping ALTER TABLE.")

# 2. Fetch all products
cursor.execute("SELECT id, title, description FROM products")
products = cursor.fetchall()

# 3. For each product, get the global type from ChatGPT and update the row
for prod_id, title, description in tqdm(products):
    prompt = (
        f"Given the following product information, return a single word for its global type.It should be only one of the following: "
        f"('car', 'boat', 'airplane', 'kit', 'accessory', 'part', 'other'):\nDon't recognize the kits or parts as a car, boat, or airplane. You should come to a serious conclusion"
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Global type:"
        "**IMPORTANT**: Only return one of the following: 'car', 'boat', 'airplane', 'kit', 'accessory', 'part', 'other'. And it should be in one word.**"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # or "gpt-4o" if you have access
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        global_type = response.choices[0].message.content.strip().split()[0]
        cursor.execute(
            "UPDATE products SET global_type = ? WHERE id = ?",
            (global_type, prod_id)
        )
        conn.commit()
    except Exception as e:
        print(f"Error processing product {prod_id}: {e}")

conn.close()

# import sqlite3
# import openai
# import os
# from dotenv import load_dotenv

# load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")

# conn = sqlite3.connect("data/inventory.db")
# cursor = conn.cursor()

# # 1. Find suspect products
# cursor.execute("""
#     SELECT id, title, description FROM products
#     WHERE global_type = 'car'
    
# """)
# suspect_products = cursor.fetchall()

# # 2. Reclassify with LLM
# for prod_id, title, description in suspect_products:
#     prompt = (
#         "The following product is currently classified as a 'car', but it may actually be a kit, accessory, or something else.\n"
#         f"Title: {title}\n"
#         f"Description: {description}\n"
#         "Please return the most accurate global type for this product (choose from: car, kit, accessory, part, other) in one word. Only in one word"
#     )
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o",  # or "gpt-4o"
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=5,
#             temperature=0
#         )
#         new_type = response.choices[0].message.content.strip().split()[0].lower()
#         cursor.execute(
#             "UPDATE products SET global_type = ? WHERE id = ?",
#             (new_type, prod_id)
#         )
#         conn.commit()
#         print(f"Updated product {prod_id} to {new_type}")
#     except Exception as e:
#         print(f"Error processing product {prod_id}: {e}")

# conn.close()
