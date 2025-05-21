import sqlite3
import json
import os

def clean_to_int(value_str):
    if value_str is None:
        return None
    try:
        return int(value_str)
    except (ValueError, TypeError):
        # print(f"Warning: Could not convert '{value_str}' to int.")
        return None

def clean_price_to_cents(price_str):
    if price_str is None:
        return None
    try:
        if isinstance(price_str, (int, float)):
            return int(float(price_str) * 100)
        price_val_str = str(price_str).lower().replace("usd", "").strip()
        if not price_val_str: return None
        return int(float(price_val_str) * 100)
    except (ValueError, TypeError):
        # print(f"Warning: Could not convert price '{price_str}' to cents.")
        return None

def clean_weight_to_int(weight_str):
    if weight_str is None:
        return None
    try:
        if isinstance(weight_str, (int, float)):
             return int(float(weight_str))
        weight_val_str = str(weight_str).lower().replace("lb", "").strip()
        if not weight_val_str: return None
        return int(float(weight_val_str)) # This will truncate
    except (ValueError, TypeError):
        # print(f"Warning: Could not convert weight '{weight_str}' to int.")
        return None

def create_and_upsert_data():
    db_dir = "data"
    db_path = os.path.join(db_dir, "inventory.db")
    products_json_path = os.path.join(db_dir, "row_data", "products.json")
    parts_json_path = os.path.join(db_dir, "row_data", "parts.json")

    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create products table
    products_table_sql = """
    CREATE TABLE IF NOT EXISTS products (
        sku TEXT,
        parent_sku TEXT,
        date_created TEXT,
        view_count INTEGER,
        total_sold INTEGER,
        price_range TEXT,
        id TEXT PRIMARY KEY,
        title TEXT,
        description TEXT,
        link TEXT,
        image_link TEXT,
        price_in_cents INTEGER,
        availability TEXT,
        brand TEXT,
        item_group_id TEXT,
        mpn TEXT,
        shipping_weight_as_int INTEGER,
        product_category TEXT
    );
    """
    cursor.execute(products_table_sql)
    print("Table 'products' created or already exists.")

    # Create parts table
    parts_table_sql = """
    CREATE TABLE IF NOT EXISTS parts (
        id TEXT PRIMARY KEY,
        item_group_id TEXT,
        sku TEXT,
        part_category TEXT,
        part_description TEXT,
        part_parentsku_compatibility TEXT,
        part_product_group_code TEXT,
        part_type TEXT
    );
    """
    cursor.execute(parts_table_sql)
    print("Table 'parts' created or already exists.")

    # Process products.json
    print(f"Processing {products_json_path}...")
    try:
        with open(products_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get("products")
        if not isinstance(items, list):
            print(f"Warning: 'products' key not found or not a list in {products_json_path}. Skipping.")
        else:
            count = 0
            product_cols = [
                'sku', 'parent_sku', 'date_created', 'view_count', 'total_sold',
                'price_range', 'id', 'title', 'description', 'link', 'image_link',
                'price_in_cents', 'availability', 'brand', 'item_group_id', 'mpn',
                'shipping_weight_as_int', 'product_category'
            ]
            placeholders = ', '.join(['?'] * len(product_cols))
            insert_sql = f"INSERT OR REPLACE INTO products ({', '.join(product_cols)}) VALUES ({placeholders})"
            
            for item in items:
                if not isinstance(item, dict):
                    print(f"Warning: Item is not a dictionary in {products_json_path}, skipping: {item}")
                    continue
                
                record = (
                    item.get('sku'), item.get('parent_sku'), item.get('date_created'),
                    clean_to_int(item.get('view_count')), clean_to_int(item.get('total_sold')),
                    item.get('price_range'), item.get('id'), item.get('title'),
                    item.get('description'), item.get('link'), item.get('image_link'),
                    clean_price_to_cents(item.get('price')), item.get('availability'),
                    item.get('brand'), item.get('item_group_id'), item.get('mpn'),
                    clean_weight_to_int(item.get('shipping_weight')), item.get('product_category')
                )
                cursor.execute(insert_sql, record)
                count += 1
            conn.commit()
            print(f"Inserted/Replaced {count} records into 'products' table.")

    except FileNotFoundError:
        print(f"Error: File not found {products_json_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {products_json_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {products_json_path}: {e}")

    # Process parts.json
    print(f"Processing {parts_json_path}...")
    try:
        with open(parts_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = data.get("products") # Still using "products" as per original request
        if not isinstance(items, list):
            print(f"Warning: 'products' key not found or not a list in {parts_json_path}. Skipping.")
        else:
            count = 0
            part_cols = [
                'id', 'item_group_id', 'sku', 'part_category', 'part_description',
                'part_parentsku_compatibility', 'part_product_group_code', 'part_type'
            ]
            placeholders = ', '.join(['?'] * len(part_cols))
            insert_sql = f"INSERT OR REPLACE INTO parts ({', '.join(part_cols)}) VALUES ({placeholders})"

            for item in items:
                if not isinstance(item, dict):
                    print(f"Warning: Item is not a dictionary in {parts_json_path}, skipping: {item}")
                    continue
                
                record = (
                    item.get('id'), item.get('item_group_id'), item.get('sku'),
                    item.get('part_category'), item.get('part_description'),
                    item.get('part_parentsku_compatibility'),
                    item.get('part_product_group_code'), item.get('part_type')
                )
                cursor.execute(insert_sql, record)
                count += 1
            conn.commit()
            print(f"Inserted/Replaced {count} records into 'parts' table.")

    except FileNotFoundError:
        print(f"Error: File not found {parts_json_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {parts_json_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {parts_json_path}: {e}")

    conn.close()
    print(f"Database operations complete. DB is at {db_path}")

if __name__ == "__main__":
    create_and_upsert_data()
