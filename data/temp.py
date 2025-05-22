import sqlite3

# Connect to your database
conn = sqlite3.connect("data/inventory.db")
cursor = conn.cursor()


cursor.execute("SELECT DISTINCT global_type FROM products")
unique_global_types = [row[0] for row in cursor.fetchall()]
print(unique_global_types)

conn.close()
