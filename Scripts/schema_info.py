import pyodbc
import json

# Connection config
server = 'SIPL05\SQLEXPRESS2014'        # e.g., localhost\\SQLEXPRESS
database = 'RMSv1_QA'
username = 'ems'
password = 'samyak'

conn_str = f"""
DRIVER={{ODBC Driver 17 for SQL Server}};
SERVER={server};
DATABASE={database};
UID={username};
PWD={password};
"""

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

schema = {}

# Get all tables
cursor.execute("""
SELECT TABLE_SCHEMA, TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
""")

tables = cursor.fetchall()

for table_schema, table_name in tables:
    full_table = f"{table_schema}.{table_name}"
    
    # Get columns
    cursor.execute(f"""
    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{table_schema}' AND TABLE_NAME = '{table_name}'
    """)
    
    columns = cursor.fetchall()
    
    column_list = []
    for col in columns:
        column_list.append({
            "column_name": col[0],
            "data_type": col[1],
            "is_nullable": col[2],
            "max_length": col[3]
        })
    
    # Get primary keys
    cursor.execute(f"""
    SELECT k.COLUMN_NAME
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
        ON t.CONSTRAINT_NAME = k.CONSTRAINT_NAME
    WHERE t.TABLE_SCHEMA = '{table_schema}'
        AND t.TABLE_NAME = '{table_name}'
        AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
    """)
    
    pk = [row[0] for row in cursor.fetchall()]
    
    schema[full_table] = {
        "columns": column_list,
        "primary_keys": pk
    }

conn.close()

# Save to JSON file
output_file = "rms_db_schema.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=4)

print(f"Schema saved to {output_file}")