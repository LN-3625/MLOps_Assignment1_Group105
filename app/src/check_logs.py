import sqlite3
import pandas as pd

# Connect to the same database used in main.py
conn = sqlite3.connect("requests.db")

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Query all prediction logs
df = pd.read_sql_query("SELECT * FROM MedHousePrediction", conn)

# Display the logs
print(df)

