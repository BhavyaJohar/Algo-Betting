"""
Step 1: Data extraction and initial schema exploration

Connect to the SQLite database, load the key tables (matches, players, rankings),
and display row counts, column info, missing values, and date ranges for matches.
"""

import sqlite3
import datetime


def explore_table(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    print(f"\n=== Table: {table_name} ===")
    print(f"Row count: {row_count}")

    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    col_names = [col[1] for col in columns_info]
    print("Columns and types:")
    for col in columns_info:
        print(f" - {col[1]} ({col[2]})")

    print("\nSample rows:")
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    rows = cursor.fetchall()
    print(col_names)
    for row in rows:
        print(row)

    print("\nMissing values per column:")
    for col in col_names:
        cursor.execute(
            f"SELECT SUM(CASE WHEN {col} IS NULL OR {col} = '' THEN 1 ELSE 0 END) FROM {table_name}"
        )
        nulls = cursor.fetchone()[0]
        print(f" - {col}: {nulls}")

    if table_name == 'matches' and 'tourney_date' in col_names:
        cursor.execute(
            f"SELECT MIN(tourney_date), MAX(tourney_date) FROM {table_name}"
        )
        min_date, max_date = cursor.fetchone()
        try:
            min_date = datetime.datetime.strptime(min_date, '%Y%m%d').date()
            max_date = datetime.datetime.strptime(max_date, '%Y%m%d').date()
            print(f"\nDate range: {min_date} to {max_date}")
        except Exception:
            print(f"\nDate range (raw): {min_date} to {max_date}")

    return None


def main():
    db_path = 'database.sqlite'
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)

    tables = ['matches', 'players', 'rankings']
    data = {}
    for tbl in tables:
        data[tbl] = explore_table(conn, tbl)

    conn.close()


if __name__ == '__main__':
    main()