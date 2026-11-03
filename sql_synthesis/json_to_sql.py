"""
Convert tables.json(Spider - style list of {"db_id": ..., "ddls": [...], ...})
into one SQLite file per database, using the folder layout expected by the
prompt - synthesis
script:

< OUT_ROOT > / < db_id > / < db_id >.sqlite

Usage:

    python json_to_sqlite.py tables.json.. / database_synthesis / synthetic_sqlite_databases
"""

import json
import os
import sqlite3
import sys

try:
    from tqdm import tqdm  # progress bar (pip install tqdm)
except ImportError:  # works without it, just no progress bar
    def tqdm(iterable, **kwargs):
        return iterable

json_path = sys.argv[1] if len(sys.argv) > 1 else "tables.json"
out_root = sys.argv[2] if len(sys.argv) > 2 else "../database_synthesis/synthetic_sqlite_databases"

# The file is ~300 MB; json.load needs a few GB of RAM for it. That is fine on most PCs.
with open(json_path, "r", encoding="utf-8") as f:
    databases = json.load(f)

os.makedirs(out_root, exist_ok=True)

created, skipped, failed_tables = 0, 0, []

for db in tqdm(databases, desc="Building databases"):
    db_id = db["db_id"]
    folder = os.path.join(out_root, db_id)
    db_file = os.path.join(folder, db_id + ".sqlite")

    # Resume support: don't rebuild what already exists
    if os.path.exists(db_file):
        skipped += 1
        continue

    os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        for ddl in db["ddls"]:
            try:
                conn.execute(ddl)
            except sqlite3.Error as e:
                failed_tables.append((db_id, str(e), ddl[:120].replace("\n", " ")))
        conn.commit()
    finally:
        conn.close()
    created += 1

print(f"\nCreated: {created}   Skipped (already existed): {skipped}")
print(f"DDL statements that failed: {len(failed_tables)}")

if failed_tables:
    log_path = os.path.join(out_root, "_failed_ddls.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(failed_tables, f, indent=2, ensure_ascii=False)
    print(f"Details written to {log_path}")
    # Note: the log sits next to the db folders; your synthesis script's os.listdir()
    # would treat it as a db name and just print an error for it. Move or delete it
    # before running the synthesis script if you want clean output.
