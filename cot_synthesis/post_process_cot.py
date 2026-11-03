import json
import os
import re
import random
import sqlite3
import hashlib
import multiprocessing as mp
from collections import Counter

import ijson
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

# ----------------------------- config -----------------------------
INPUT_FILE = "cot_synthesis/results/cot_synthesis.json"
OUTPUT_FILE = "cot_synthesis/results/synthetic_text2sql_dataset.json"
DB_ROOT = "database_synthesis/sinthetic_sqlite_databases"  # <db_id>/<db_id>.sqlite
NUM_PROCESSES = 20
TIMEOUT = 400            # seconds per query (in-memory copy + execution)
BATCH_SIZE = 10240
MIN_VALID = 3          # lowered automatically to the number of samples if fewer were generated
MAX_TRAILING_CHARS = 300  # text allowed after the final ```sql block
MERGE_EXTERNAL_KNOWLEDGE = True  # put external knowledge in front of the question (as OmniSQL does)
KEEP_METADATA = False            # True: also keep sql_complexity / question_style in each record
# -------------------------------------------------------------------

FENCE = re.compile(r"```(?:sqlite|sql)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
LEADING_KW = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/|\s)*(\w+)", re.DOTALL)


def parse_response(response):
    """Return the final SQL block, or "" if the response has no usable final answer.

    A response is rejected if a lot of text follows the last code block: that means the
    model was cut off (token limit) and the last block is just a schema/reference quoted
    inside the reasoning, not the answer.
    """
    matches = list(FENCE.finditer(response))
    if not matches:
        return ""
    last = matches[-1]
    if len(response) - last.end() > MAX_TRAILING_CHARS:
        return ""
    return last.group(1).strip()


def first_keyword(sql):
    m = LEADING_KW.match(sql)
    return m.group(1).lower() if m else ""


def state_signature(conn):
    """Hash of the whole database content, used to compare non-SELECT statements."""
    h = hashlib.md5()
    for line in conn.iterdump():
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def execute_sql(task):
    """Run one query without ever modifying the real database.

    Returns (idx, result, error). `result` is a frozenset of rows for statements that
    return rows, otherwise a hash of the resulting database state (INSERT/UPDATE/DELETE/DDL).
    """
    idx, db_file, sql = task
    if not sql.strip():
        return idx, None, "empty_sql"
    if not os.path.exists(db_file):
        return idx, None, "db_not_found"

    src = conn = None
    try:
        if first_keyword(sql) == "select":
            # plain SELECT: read-only connection is enough
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute(sql)
            return idx, frozenset(cur.fetchall()), None

        # everything else runs on a private in-memory copy
        src = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        conn = sqlite3.connect(":memory:")
        src.backup(conn)
        src.close()
        src = None

        cur = conn.cursor()
        try:
            cur.execute(sql)
            if cur.description:
                result = frozenset(cur.fetchall())
            else:
                conn.commit()
                result = state_signature(conn)
        except (sqlite3.ProgrammingError, sqlite3.Warning):
            # several statements in one block (CREATE ...; INSERT ...;)
            conn.executescript(sql)
            result = state_signature(conn)
        return idx, result, None
    except Exception as e:
        return idx, None, f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        for c in (src, conn):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass


def run_task(task):
    try:
        return func_timeout(TIMEOUT, execute_sql, args=(task,))
    except FunctionTimedOut:
        return task[0], None, "timeout"
    except Exception as e:
        return task[0], None, f"{type(e).__name__}: {str(e)[:100]}"


def build_record(rec, cot, sql):
    """Final <database, question, SQL query, CoT solution> record."""
    question = rec["question"].strip()
    ek = (rec.get("external_knowledge") or "").strip()
    if MERGE_EXTERNAL_KNOWLEDGE and ek:
        question = ek + "\n" + question
    out = {
        "database": rec["db_id"],
        "question": question,
        "sql": sql,
        "cot": cot,
    }
    if KEEP_METADATA:
        out["sql_complexity"] = rec["sql_complexity"]
        out["question_style"] = rec["question_style"]
    return out


def resolve_db_path(db_id):
    for ext in (".sqlite", ".db"):
        p = os.path.join(DB_ROOT, db_id, db_id + ext)
        if os.path.exists(p):
            return p
    return os.path.join(DB_ROOT, db_id, db_id + ".sqlite")


def load_json_file(file):
    dataset = []
    with open(file, "r", encoding="utf-8") as f:
        for obj in ijson.items(f, "item"):
            dataset.append(obj)
    return dataset


if __name__ == "__main__":
    results = load_json_file(INPUT_FILE)
    print("records:", len(results))
    print("samples per record:", Counter(len(r["responses"]) for r in results))

    filtered = 0
    final_dataset = []
    errors = Counter()
    sql_type_total, sql_type_kept = Counter(), Counter()

    with mp.Pool(processes=NUM_PROCESSES) as pool:
        for start in tqdm(range(0, len(results), BATCH_SIZE)):
            batch = results[start:start + BATCH_SIZE]

            tasks, owners = [], []  # owners[i] = (record index in batch, response index)
            parsed = []
            for r_idx, rec in enumerate(batch):
                db_file = resolve_db_path(rec["db_id"])
                sqls = [parse_response(resp) for resp in rec["responses"]]
                parsed.append(sqls)
                for s_idx, sql in enumerate(sqls):
                    tasks.append((len(tasks), db_file, sql))
                    owners.append((r_idx, s_idx))

            outputs = list(pool.imap(run_task, tasks, chunksize=16))  # keeps input order

            per_record = [dict() for _ in batch]  # execution result -> list of response idx
            for (_, result, err), (r_idx, s_idx) in zip(outputs, owners):
                if err is not None:
                    errors[err.split(":")[0] if err in ("empty_sql", "db_not_found", "timeout") else err] += 1
                    continue
                per_record[r_idx].setdefault(result, []).append(s_idx)

            for r_idx, rec in enumerate(batch):
                t = first_keyword(rec["sql"]).upper() or "UNKNOWN"
                sql_type_total[t] += 1

                voting = per_record[r_idx]
                valid = sum(len(v) for v in voting.values())
                min_valid = min(MIN_VALID, len(rec["responses"]))
                if valid < min_valid:
                    filtered += 1
                    continue

                best = max(voting, key=lambda k: len(voting[k]))
                s_idx = random.choice(voting[best])
                final_cot = rec["responses"][s_idx]

                final_dataset.append(build_record(rec, final_cot, parsed[r_idx][s_idx]))
                sql_type_kept[t] += 1

    print("filtered out:", filtered)
    print("kept:", len(final_dataset))
    print("kept / total by type of source SQL:")
    for t, n in sql_type_total.most_common():
        print(f"  {t}: {sql_type_kept[t]}/{n}")
    print("top failure reasons:")
    for e, n in errors.most_common(10):
        print(f"  {n:6d}  {e}")

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)