import sqlite3
import pandas as pd

# Connect to the same database used in main.py
conn = sqlite3.connect("requests.db")

# Query all prediction logs
df = pd.read_sql_query("SELECT * FROM MedHouseVal_prediction", conn)

# Display the logs
print(df)