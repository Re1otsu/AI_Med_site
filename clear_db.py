import os
import sqlite3
from app.db import DB_PATH

print("Используемая база:", DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("DELETE FROM history;")
cur.execute("DELETE FROM patients;")
conn.commit()
conn.close()

print("Готово! База очищена.")
