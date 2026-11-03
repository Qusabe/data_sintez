# =====================================================================================
# CHANGE LOG (changed/added places are tagged "# [CHANGE n]"):
#   [CHANGE 1] SELECT-only filter removed. filter_select_sqls() is no longer called; every
#              statement type (SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ...) stays.
#              KEEP_STATEMENT_TYPES lets you restrict types again if you ever want to.
#   [CHANGE 2] get_statement_type(): reliable detection of the statement type (also handles
#              comments and WITH ... INSERT/UPDATE/DELETE). Saved as "statement_type".
#   [CHANGE 3] execute_sql() rewritten so non-SELECT statements count as valid. The original
#              crashed on len(cursor.description) (it is None for INSERT/UPDATE/DELETE/DDL)
#              and the statement was silently dropped in the "remove timeout sqls" stage.
#   [CHANGE 4] Safety: the original databases are opened READ-ONLY. Statements that modify
#              data/schema run on an in-memory COPY, so the real .sqlite files can never be
#              changed (and parallel workers can't block each other with write locks).
#              Statements that cannot be run in a transaction (VACUUM, ATTACH, DETACH,
#              BEGIN/COMMIT/ROLLBACK/SAVEPOINT/RELEASE) are only compiled, never executed.
#   [CHANGE 5] Output records get new fields: "statement_type", "requested_statement_type"
#              (what synthesize_sql.py asked for), "affected_rows" (for INSERT/UPDATE/DELETE).
#              Existing fields keep their meaning; for non-SELECT, column_count is 0.
#   [CHANGE 6] Per-statement-type counts are printed after every filtering stage, so you can
#              see that nothing is dropped because of its type.
#   [CHANGE 7] Small robustness fixes (empty-SQL return value, empty-result guard, missing
#              database no longer silently creates an empty .sqlite file).
# =====================================================================================

import json
import sqlite3
import os
import sys
import re
import time
from pathlib import Path  # [CHANGE 4]

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
import ijson

# [CHANGE 1] None = keep every statement type. Example of restricting: {"SELECT", "UPDATE"}
KEEP_STATEMENT_TYPES = None

# [CHANGE 2] ---- statement type detection ------------------------------------------------
# Tokenizer that understands comments and quoted strings/identifiers, so that e.g. a
# comment before the statement or the word "update" inside a string cannot fool the detection.
_TOKEN_RE = re.compile(
    r"""
      (?P<space>\s+)
    | (?P<comment>--[^\n]*|/\*.*?\*/)
    | (?P<string>'(?:[^']|'')*'|"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\])
    | (?P<word>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<punct>[(),;])
    | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)

_MAIN_KEYWORDS = {"select", "insert", "replace", "update", "delete"}

_TYPE_MAP = {
    "select": "SELECT", "values": "SELECT",
    "insert": "INSERT", "replace": "INSERT",
    "update": "UPDATE", "delete": "DELETE",
    "create": "CREATE", "drop": "DROP", "alter": "ALTER",
    "pragma": "PRAGMA", "vacuum": "VACUUM",
    "attach": "ATTACH", "detach": "DETACH",
    "begin": "BEGIN", "commit": "COMMIT", "end": "COMMIT", "rollback": "ROLLBACK",
    "savepoint": "SAVEPOINT", "release": "RELEASE",
    "reindex": "REINDEX", "analyze": "ANALYZE", "explain": "EXPLAIN",
}


def get_statement_type(sql):
    '''Return SELECT / INSERT / UPDATE / DELETE / CREATE / DROP / ALTER / PRAGMA / ... / OTHER'''
    first = None
    depth = 0
    for m in _TOKEN_RE.finditer(sql):
        kind = m.lastgroup
        if kind in ("space", "comment", "string"):
            continue
        tok = m.group()
        if first is None:
            if kind != "word":
                return "OTHER"
            first = tok.lower()
            if first != "with":
                break
            continue
        # we are inside "WITH ..." : the real statement is the first main keyword outside parentheses
        if kind == "punct":
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
        elif kind == "word" and depth == 0 and tok.lower() in _MAIN_KEYWORDS:
            first = tok.lower()
            break
    if first is None or first == "with":
        return "OTHER"
    return _TYPE_MAP.get(first, "OTHER")


# [CHANGE 4] ---- how each type is executed ---------------------------------------------
READ_ONLY_TYPES = {"SELECT", "EXPLAIN"}  # run directly on a read-only connection
COMPILE_ONLY_TYPES = {  # cannot be run inside a transaction -> only compiled with EXPLAIN
    "VACUUM", "ATTACH", "DETACH", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
}


# every other type (INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, PRAGMA, REINDEX, ANALYZE, ...)
# is executed on an in-memory copy of the database inside BEGIN ... ROLLBACK.

def connect_read_only(db_path):
    # mode=ro: physically cannot modify the file, and (unlike plain connect) does NOT create
    # an empty database when the path is wrong  [CHANGE 7]
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def copy_to_memory(db_path):
    src_conn = connect_read_only(db_path)
    try:
        mem_conn = sqlite3.connect(":memory:")
        src_conn.backup(mem_conn)
    finally:
        src_conn.close()
    return mem_conn


def execute_sql(sql, db_path, statement_type=None):
    '''
    Returns (execution_result, column_count, affected_rows); (None, None, None) if the statement failed.
    [CHANGE 3] Non-SELECT statements are valid: execution_result == [], column_count == 0,
    affected_rows == number of changed rows (INSERT/UPDATE/DELETE).
    '''
    if sql.strip() == "":
        return None, None, None  # [CHANGE 7] was `return None`, which broke tuple unpacking

    if statement_type is None:
        statement_type = get_statement_type(sql)

    conn = None
    try:
        if statement_type in COMPILE_ONLY_TYPES:
            # compile only (never executed) - checks the syntax without any side effects
            conn = connect_read_only(db_path)
            conn.execute("EXPLAIN " + sql).fetchall()
            return [], 0, 0

        if statement_type in READ_ONLY_TYPES:
            conn = connect_read_only(db_path)
            conn.isolation_level = None  # we control transactions ourselves
            cursor = conn.cursor()
            cursor.execute(sql)
        else:
            conn = copy_to_memory(db_path)  # the real database file is never touched
            conn.isolation_level = None
            cursor = conn.cursor()
            cursor.execute("BEGIN")
            cursor.execute(sql)

        execution_result = cursor.fetchall()
        if cursor.description is not None:  # statement returns columns (SELECT, PRAGMA, ... RETURNING)
            column_count = len(cursor.description)
            affected_rows = 0
        else:  # INSERT / UPDATE / DELETE / DDL
            column_count = 0
            affected_rows = max(cursor.rowcount, 0)  # rowcount is -1 for DDL
        return execution_result, column_count, affected_rows
    except Exception as e:
        # print(f"An error occurred: {e}")
        return None, None, None
    finally:
        if conn is not None:
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")  # undo everything (also protects the in-memory copy's state)
            except Exception:
                pass
            conn.close()


# [CHANGE 3/5] extra arguments: statement_type, requested_statement_type
def execute_wrapper(sample_idx, db_id, sql, complexity, timeout, db_dir, statement_type=None,
                    requested_statement_type=None):
    try:
        execution_result, column_count, affected_rows = func_timeout(
            timeout, execute_sql,
            args=(sql, os.path.join(db_dir, db_id, db_id + ".sqlite"), statement_type)
        )
        if execution_result is None or column_count is None:
            return [sample_idx, db_id, sql, complexity, 0, 0, 0, 0, statement_type, requested_statement_type]
        else:
            # "rows": number of result rows for statements that return rows (SELECT),
            # number of changed rows for INSERT/UPDATE/DELETE
            rows = len(execution_result) if column_count > 0 else affected_rows
            return [sample_idx, db_id, sql, complexity, 1, column_count, rows, affected_rows, statement_type,
                    requested_statement_type]
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        return [sample_idx, db_id, sql, complexity, 0, 0, 0, 0, statement_type, requested_statement_type]
    except Exception as e:
        return [sample_idx, db_id, sql, complexity, 0, 0, 0, 0, statement_type, requested_statement_type]


def execute_callback(result):
    sample_idx, db_id, sql, complexity, valid_flag, column_count, rows, affected_rows, statement_type, requested_statement_type = result
    if valid_flag == 1:
        no_timeout_synthesized_sqls.append(
            {
                "db_id": db_id, "sql": sql, "column_count": column_count, "rows": rows, "complexity": complexity,
                "statement_type": statement_type,  # [CHANGE 5]
                "requested_statement_type": requested_statement_type,  # [CHANGE 5]
                "affected_rows": affected_rows,  # [CHANGE 5]
            }
        )
    # print("Done:", sample_idx)


def remove_timeout_sqls_parallel(synthesized_sqls, db_dir, num_cpus=20, timeout=1):
    '''Execute the sqls in parallel'''
    parallel_batch_size = 10240
    batches = [synthesized_sqls[i: i + parallel_batch_size] for i in
               range(0, len(synthesized_sqls), parallel_batch_size)]

    assert len(synthesized_sqls) == sum([len(batch_sqls) for batch_sqls in batches])

    for batch_idx, batch_sqls in enumerate(batches):
        print(f"execution process: {batch_idx + 1}/{len(batches)}")
        pool = mp.Pool(processes=num_cpus)
        for sample_idx, sql_info in enumerate(batch_sqls):
            pool.apply_async(
                execute_wrapper,
                args=(sample_idx, sql_info["db_id"], sql_info["sql"], sql_info["complexity"], timeout, db_dir,
                      sql_info.get("statement_type"), sql_info.get("requested_statement_type")),  # [CHANGE 5]
                callback=execute_callback
            )
        pool.close()
        pool.join()
        time.sleep(10)


def analyze_complexity(results):
    complexity2num = dict()
    for res in results:
        complexity = res["complexity"]
        if complexity in complexity2num:
            complexity2num[complexity] += 1
        else:
            complexity2num[complexity] = 1
    print(complexity2num)


# [CHANGE 6] how many statements of each type are left after a stage
def analyze_statement_types(results):
    type2num = dict()
    for res in results:
        statement_type = res.get("statement_type") or get_statement_type(res["sql"])
        type2num[statement_type] = type2num.get(statement_type, 0) + 1
    print("statement types:", dict(sorted(type2num.items(), key=lambda kv: -kv[1])))


def analyze_column_count(results):
    column_count2num = dict()
    for res in results:
        column_count = res["column_count"]
        if column_count in column_count2num:
            column_count2num[column_count] += 1
        else:
            column_count2num[column_count] = 1
    print(column_count2num)


def analyze_advanced_functions(results):
    function2num = dict()
    functions = json.load(open("sql_synthesis/prompt_templates/sqlite_funcs.json"))
    functions = [func_desc.split("(")[0] for func_desc in functions]

    for res in results:
        sql = res["sql"]
        for function in functions:
            if function.lower() + "(" in sql.lower():
                if function in function2num:
                    function2num[function] += 1
                else:
                    function2num[function] = 1

    print(function2num)


def analyze_used_tables_num(synthesized_sqls, db_id2table_names):
    used_tables_num2count = dict()
    for sql_info in tqdm(synthesized_sqls):
        table_names_in_db = db_id2table_names[sql_info["db_id"]]
        sql = sql_info["sql"]
        if sql.endswith(";"):
            sql = sql[:-1]
        sql_tokens = sql.strip().lower().split()
        # print(table_names_in_db)
        # print(sql_tokens)

        used_tables = set()

        for table_name in table_names_in_db:
            if table_name.lower() in sql_tokens:
                used_tables.add(table_name.lower())

        used_tables_num = len(used_tables)
        # print(used_tables)
        # print(used_tables_num)
        # print("------------------------------------------")
        if used_tables_num in used_tables_num2count:
            used_tables_num2count[used_tables_num] += 1
        else:
            used_tables_num2count[used_tables_num] = 1

    print(used_tables_num2count)


def filter_executable_sqls(synthesized_sqls, db_dir):
    executable_sqls = []
    for sql_info in tqdm(synthesized_sqls):
        db_path = os.path.join(db_dir, sql_info["db_id"], sql_info["db_id"] + ".sqlite")
        # [CHANGE 3] works for every statement type; it only compiles the statement, nothing is executed
        query_plan, _, _ = execute_sql("EXPLAIN QUERY PLAN " + sql_info["sql"], db_path)
        if query_plan is not None:
            sql_info["query_plan"] = str(query_plan)
            executable_sqls.append(sql_info)
    return executable_sqls


def filter_select_sqls(synthesized_sqls):
    '''
        remain SELECT-type queries
        [CHANGE 1] NOT USED ANY MORE - kept only for reference. Use filter_statement_types() instead.
    '''
    select_sqls = []
    for sql_info in tqdm(synthesized_sqls):
        # remove comments
        sql_wo_comments = re.sub(r'/\*.*?\*/', '', sql_info["sql"], flags=re.DOTALL)
        sql_wo_comments = re.sub(r'--.*', '', sql_wo_comments)
        sql_wo_comments = sql_wo_comments.strip()

        if sql_wo_comments.lower().startswith("select") or \
                sql_wo_comments.lower().startswith("with"):
            select_sqls.append(sql_info)
    return select_sqls


# [CHANGE 1] replacement for filter_select_sqls(): keeps everything unless KEEP_STATEMENT_TYPES is set
def filter_statement_types(synthesized_sqls, keep_types=None):
    if keep_types is None:
        return synthesized_sqls
    return [sql_info for sql_info in synthesized_sqls if sql_info["statement_type"] in keep_types]


def dedup_using_query_plan(synthesized_sqls):
    # NOTE [CHANGE 1]: do not enable this for non-SELECT statements. INSERT/DDL statements
    # have an (almost) empty query plan, so they would all collapse into a few samples.
    unique_plans = set()
    deduped_sqls = []
    for sql_info in tqdm(synthesized_sqls):
        query_plan = sql_info["query_plan"]
        if query_plan not in unique_plans:
            unique_plans.add(query_plan)
            deduped_sqls.append(sql_info)
    return deduped_sqls


def obtain_sql_template(sql):
    # Handles single and double quoted strings, numbers, NULL, TRUE, FALSE
    pattern = r"""
        (?<!\w)'(?:\\.|[^'])*' |  # single quoted strings
        (?<!\w)"(?:\\.|[^"])*" |  # double quoted strings
        (?<!\w)-?\b\d+(\.\d+)?([eE][-+]?\d+)?\b | # numbers with scientific notation
        \bNULL\b |                # NULL
        \bTRUE\b |                # TRUE
        \bFALSE\b                 # FALSE
    """

    # replace values with a special token <value>
    template = re.sub(pattern, "<value>", sql, flags=re.IGNORECASE | re.VERBOSE)
    template = template.lower().replace("\n", " ").strip()

    # Replace multiple spaces with a single space
    template = re.sub(r'\s+', ' ', template)

    return template


def dedup_using_query_template(synthesized_sqls):
    unique_templates = set()
    deduped_sqls = []
    for sql_info in tqdm(synthesized_sqls):
        template = obtain_sql_template(sql_info["sql"])
        if template not in unique_templates:
            unique_templates.add(template)
            deduped_sqls.append(sql_info)
    return deduped_sqls


def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"

    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""


def obtain_db_id2table_names(results, db_dir):
    db_ids = list(set([res["db_id"] for res in results]))
    print("len(db_ids):", len(db_ids))
    db_id2table_names = dict()
    for db_id in db_ids:
        results, _, _ = execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table';",
            os.path.join(db_dir, db_id, db_id + ".sqlite")
        )
        table_names = [res[0] for res in (results or [])]  # [CHANGE 7] guard against failed lookups
        db_id2table_names[db_id] = table_names
    return db_id2table_names


def load_json_file(file):
    dataset = []
    with open(file, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        for obj in tqdm(objects):
            dataset.append(obj)
    return dataset


if __name__ == "__main__":
    synthesized_sqls = []
    db_dir = "database_synthesis/sinthetic_sqlite_databases"
    llm_responses = load_json_file("sql_synthesis/results/sql_synthesis.json")

    for llm_response in tqdm(llm_responses):
        sql = parse_response(llm_response["response"])
        if sql == "":
            continue
        synthesized_sqls.append(
            {
                "db_id": llm_response["db_id"][:-3] if llm_response["db_id"].endswith(".db") else llm_response["db_id"],
                "sql": sql,
                "complexity": llm_response["prompt"].split("Ensure the SQL query matches the ")[1].split(
                    " level, defined as follows:")[0],
                "statement_type": get_statement_type(sql),  # [CHANGE 2] what the SQL really is
                "requested_statement_type": llm_response.get("statement_type"),  # [CHANGE 5] what the prompt asked for
            }
        )

    print("original sql num:", len(synthesized_sqls))
    # analyze_complexity(synthesized_sqls)
    # analyze_advanced_functions(synthesized_sqls)

    analyze_statement_types(synthesized_sqls)  # [CHANGE 6]

    # [CHANGE 1] the SELECT-only filter was here. Now all statement types are kept
    # (unless you set KEEP_STATEMENT_TYPES at the top of the file).
    synthesized_sqls = filter_statement_types(synthesized_sqls, KEEP_STATEMENT_TYPES)
    print("sql num after statement-type filter (keeps all types by default):", len(synthesized_sqls))
    # analyze_complexity(synthesized_sqls)
    # analyze_advanced_functions(synthesized_sqls)

    # remove sqls with syntax errors
    synthesized_sqls = filter_executable_sqls(synthesized_sqls, db_dir)
    print("sql num after removing syntax-error sqls:", len(synthesized_sqls))
    analyze_statement_types(synthesized_sqls)  # [CHANGE 6]
    # analyze_complexity(synthesized_sqls)
    # analyze_advanced_functions(synthesized_sqls)

    # # perform deduplication according to the query plan
    # synthesized_sqls = dedup_using_query_plan(synthesized_sqls)
    # print("sql num after deduplication (query plan level):", len(synthesized_sqls))
    # print(synthesized_sqls[0].keys())
    # # analyze_complexity(synthesized_sqls)
    # # analyze_advanced_functions(synthesized_sqls)

    # remove timeout sqls
    no_timeout_synthesized_sqls = mp.Manager().list()
    remove_timeout_sqls_parallel(synthesized_sqls, db_dir, 10, 2)
    synthesized_sqls = list(no_timeout_synthesized_sqls)
    print("sql num after removing timeout sqls:", len(synthesized_sqls))
    if synthesized_sqls:  # [CHANGE 7] avoid IndexError on an empty list
        print(synthesized_sqls[0].keys())
    analyze_statement_types(synthesized_sqls)  # [CHANGE 6]
    # analyze_complexity(synthesized_sqls)
    analyze_column_count(synthesized_sqls)
    # analyze_advanced_functions(synthesized_sqls)

    # perform deduplication according to the query template
    synthesized_sqls = dedup_using_query_template(synthesized_sqls)
    print("sql num after deduplication (tempalte level):", len(synthesized_sqls))
    analyze_statement_types(synthesized_sqls)  # [CHANGE 6]
    # analyze_complexity(synthesized_sqls)
    analyze_column_count(synthesized_sqls)
    # analyze_advanced_functions(synthesized_sqls)

    # analyze the number of used tables
    analyze_used_tables_num(
        synthesized_sqls,
        obtain_db_id2table_names(synthesized_sqls, db_dir)
    )

    with open("sql_synthesis/results/synthetic_sqls.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(synthesized_sqls, indent=2, ensure_ascii=False))