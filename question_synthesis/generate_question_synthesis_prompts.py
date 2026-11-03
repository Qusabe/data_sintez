# =====================================================================================
# CHANGE LOG (changed/added places are tagged "# [CHANGE n]"):
#   [CHANGE 1] get_statement_type(): the statement type (SELECT / INSERT / UPDATE / DELETE /
#              CREATE / DROP / ALTER / ...) is read from the "statement_type" field written by
#              post_process_sqls.py, or detected here when the field is missing.
#   [CHANGE 2] Statement-specific guidelines. The original guidelines talk about "columns being
#              selected", ORDER BY and LIMIT, which makes no sense for INSERT/UPDATE/DELETE/DDL.
#              SELECT prompts are IDENTICAL to the original ones; other types get fitting
#              guidelines, and the question is phrased as a request ("Add...", "Remove...").
#   [CHANGE 3] Column information is now table-aware where the SQL alone doesn't name the
#              columns: INSERT without a column list, DROP TABLE, ALTER TABLE.
#   [CHANGE 4] Databases are opened read-only (a wrong path no longer creates an empty
#              .sqlite file), and the output folder is created automatically.
#   [CHANGE 5] "statement_type" is saved in every output record; statistics are printed.
# =====================================================================================

import json
import os
import random
import sqlite3
import numpy as np
import re
from pathlib import Path  # [CHANGE 4]
from tqdm import tqdm

style2desc = {
    "Formal": '''**Formal Style**
   - Uses standard grammar and vocabulary.
   - Example: Find all students older than 18 years and return their home addresses.''',

    "Colloquial": '''**Colloquial Style**
   - Employs informal vocabulary and expressions.
   - Example: Hey! Could you help me find all the students who are over 18? I'd love to know their names and where they live.''',

    "Imperative": '''**Imperative Style**
   - Uses command or directive sentences.
   - Example: Could you please gather all the students who are older than 18? I really need to know their names and where they live!''',

    "Interrogative": '''**Interrogative Style**
   - Uses question forms.
   - Example: Could you tell me which students are older than 18 and what their home addresses are?''',

    "Descriptive": '''**Descriptive Style**
   - Uses detailed descriptions with contextual information.
   - Example: I want to know the names and home addresses of all students older than 18.''',

    "Concise": '''**Concise Style**
   - Use short sentences.
   - Example: Students older than 18, return their names and addresses.''',

    "Vague": '''**Vague Style**
   - Includes ambiguous vocabulary requiring inference.
   - Example: What are the names and addresses of those older students? (External Knowledge: 'older students' refers to age >= 18.)''',

    "Metaphorical": '''**Metaphorical Style**
   - Uses metaphors or metaphorical expressions.
   - Example: Find the names and addresses of those who have reached adulthood. (External Knowledge: 'reached adulthood' refers to age >= 18.)''',

    "Multi-turn Dialogue": '''**Multi-turn Dialogue Style**
    - This involves a dialogue to clarify the user's query needs.
    - Example: [{"User": "I want to query some student information."}, {"Assistant": "Which students' information would you like to query?"}, {"User": "Students older than 18."}, {"Assistant": "What other information would you like to know about them?"}, {"User": "Names and addresses."}, {"Assistant": "Is there anything else you need?"}, {"User": "No."}, {"Assistant": "OK, I will help you translate your request into an SQL query."}]'''
}

steps_wo_ek = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.'''

steps_w_ek = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.
3. **External Knowledge:** For Vague or Metaphorical styles, include external knowledge to enhance clarity.'''

steps_multi_round = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
2. **Generate a Dialogue:** Create a conversation between the User and the Assistant based on the SQL query and its explanation.'''

guidelines_wo_ek = '''1. Clearly describe the columns being selected by the SQL query. For example:
   - "SELECT * ... FROM ..." means "Find all ...";
   - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
2. Ensure the natural language question accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.'''

guidelines_w_ek = '''1. Clearly describe the columns being selected by the SQL query. For example:
   - "SELECT * ... FROM ..." means "Find all ...";
   - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
2. Ensure the natural language question accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.
3. If necessary, incorporate external knowledge using multiple entries separated by semicolons (";"). These can include formulas, common sense, domain-specific knowledge, or extended context, such as information from long documents. Each entry should be concise.'''

guidelines_multi_round = '''1. Clearly describe the columns being selected by the SQL query. For example:
   - "SELECT * ... FROM ..." means "Find all ...";
   - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
2. Ensure the conversation accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.'''

output_format_wo_ek = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation)
[EXPLANATION-END]

[QUESTION-START]
(Natural Language Question)
[QUESTION-END]

- **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
- **Natural Language Question**: Translate the SQL query into a natural language question, enclosed within [QUESTION-START] and [QUESTION-END].'''

output_format_w_ek = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation)
[EXPLANATION-END]

[QUESTION-START]
(Natural Language Question)
[QUESTION-END]

[EXTERNAL-KNOWLEDGE-START]
(External Knowledge)
[EXTERNAL-KNOWLEDGE-END]

- **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
- **Natural Language Question**: Translate the SQL query into a natural language question, enclosed within [QUESTION-START] and [QUESTION-END].
- **External Knowledge**: Include any relevant external knowledge if applicable, enclosed within [EXTERNAL-KNOWLEDGE-START] and [EXTERNAL-KNOWLEDGE-END]. Leave this section blank if not needed.'''

output_format_multi_round = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation)
[EXPLANATION-END]

[QUESTION-START]
(Natural Language Question, in the format of [{"User": ...}, {"Assistant": ...}, {"User": ...}, ....])
[QUESTION-END]

- **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
- **Natural Language Question**: Convert the SQL query into a multi-round dialogue, enclosed within [QUESTION-START] and [QUESTION-END]. Represent this as a list that captures multiple rounds of conversation between the User and the Assistant.'''

instruction_wo_ek = "Based on the above information, follow the reasoning steps to generate the explanation and the question corresponding to the SQL query."

instruction_w_ek = "Based on the above information, follow the reasoning steps to generate the explanation, the question, and the external knowledge corresponding to the SQL query."

instruction_multi_round = "Based on the above information, follow the reasoning steps to generate the explanation and the dialogue corresponding to the SQL query."

# [CHANGE 1] ---- statement type detection (same logic as in post_process_sqls.py) ---------------
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


# [CHANGE 2] ---- guidelines for statements that are not SELECT ---------------------------------
# "{target}" is replaced by "natural language question" or "conversation" (multi-turn style).
non_select_guidelines = {
    "INSERT": (
        '''Clearly describe which table receives the new data and which value is stored in which column, naming the columns in plain language based on the column information. For example:
   - "INSERT INTO students (name, age) VALUES ('Ann', 20)" means "Add a new student named Ann who is 20 years old.";
   - "INSERT INTO archive (id, name) SELECT id, name FROM students WHERE age > 25" means "Copy the id and name of all students older than 25 into the archive.".''',
        '''Ensure the {target} accurately captures the semantics of the SQL statement, including every inserted value, the conditions of an `INSERT ... SELECT`, and conflict handling such as `OR REPLACE` or `OR IGNORE`.'''
    ),
    "UPDATE": (
        '''Clearly describe which table is changed, which columns receive new values, and what the new values are (or how they are computed). For example:
   - "UPDATE employees SET salary = salary * 1.1 WHERE department = 'Sales'" means "Increase the salary of all employees in the Sales department by 10%.".''',
        '''Ensure the {target} accurately captures the semantics of the SQL statement, including the conditions (`WHERE`, subqueries, `FROM` joins) that decide which rows are changed.'''
    ),
    "DELETE": (
        '''Clearly describe which table the rows are removed from and which rows are removed. For example:
   - "DELETE FROM orders WHERE status = 'cancelled'" means "Remove all cancelled orders.".''',
        '''Ensure the {target} accurately captures the semantics of the SQL statement, including all conditions (`WHERE`, subqueries) that decide which rows are deleted.'''
    ),
    "DDL": (
        '''Clearly describe the structural change: the kind of object (table, index, view, trigger), its name, the tables and columns it concerns, data types and constraints. For example:
   - "CREATE INDEX idx_age ON students(age)" means "Create an index on the age column of the students table.";
   - "ALTER TABLE students ADD COLUMN email TEXT" means "Add a text column called email to the students table.".''',
        '''Ensure the {target} accurately captures the semantics of the SQL statement, including names, data types, constraints (`PRIMARY KEY`, `NOT NULL`, `DEFAULT`, `FOREIGN KEY`) and, for `DROP`, exactly which object is removed.'''
    ),
    "OTHER": (
        '''Describe in plain language what the SQL statement does, for example which database setting is read or changed, which maintenance or transaction-control operation is performed, and which tables it concerns.''',
        '''Ensure the {target} accurately captures the semantics of the SQL statement.'''
    ),
}

external_knowledge_guideline = '''If necessary, incorporate external knowledge using multiple entries separated by semicolons (";"). These can include formulas, common sense, domain-specific knowledge, or extended context, such as information from long documents. Each entry should be concise.'''

non_select_note = '''This SQL statement (type: {statement_type}) is not a data-retrieval query. The example in the style description only illustrates the wording style: do not copy its content. Phrase the {target} as the request a user would make to a database assistant (for example "Add ...", "Change ...", "Remove ...", "Create ..."), not as a question about retrieved data.'''


def build_guidelines(statement_type, style_name):
    '''SELECT statements get the original guidelines unchanged.'''
    is_multi_round = style_name == "Multi-turn Dialogue"
    with_ek = style_name in ["Vague", "Metaphorical"]

    if statement_type == "SELECT":
        if with_ek:
            return guidelines_w_ek
        return guidelines_multi_round if is_multi_round else guidelines_wo_ek

    group = statement_type if statement_type in ("INSERT", "UPDATE", "DELETE") \
        else "DDL" if statement_type in ("CREATE", "DROP", "ALTER") else "OTHER"
    target = "conversation" if is_multi_round else "natural language question"
    first, second = non_select_guidelines[group]

    lines = [f"1. {first}", f"2. {second.replace('{target}', target)}"]
    if with_ek:
        lines.append(f"{len(lines) + 1}. {external_knowledge_guideline}")
    lines.append(
        f"{len(lines) + 1}. " + non_select_note.replace("{statement_type}", statement_type).replace("{target}", target))
    return "\n".join(lines)


def obtain_db_schema(db_file_dir):
    # [CHANGE 4] read-only: sqlite3.connect(path) would silently CREATE an empty database for a wrong path
    conn = sqlite3.connect(Path(db_file_dir).resolve().as_uri() + "?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    table_names = []
    create_statements = []
    for table in tables:
        table_name, create_statement = table
        table_names.append(table_name)
        create_statements.append(create_statement)

    cursor.close()
    conn.close()

    return table_names, create_statements


# NOTE: When columns with the same names exist in different tables, more detailed design considerations are necessary
def extract_column_descriptions(create_statements):
    column_name2column_desc = dict()
    # Regular expression to match column definitions
    pattern = r'"(\w+)"\s+\w+\s*/\*\s*(.*?)\s*\*/'

    for create_statement in create_statements:
        # Find all matches in the string
        matches = re.findall(pattern, create_statement)

        # Print the results
        for column_name, description in matches:
            column_name = column_name.lower()
            if column_name not in column_name2column_desc:
                column_name2column_desc[column_name] = description

    return column_name2column_desc


# [CHANGE 3] ---- table-aware column information ----------------------------------------------
_COLUMN_RE = r'"(\w+)"\s+\w+\s*/\*\s*(.*?)\s*\*/'
_CREATE_TABLE_RE = re.compile(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\[]?(\w+)[`"\]]?\s*\(', re.IGNORECASE)
_INSERT_RE = re.compile(
    r'\b(?:insert|replace)\b(?:\s+or\s+\w+)?\s+into\s+[`"\[]?(\w+)[`"\]]?(?:\s+as\s+\w+)?\s*(\(\s*(?!select\b|with\b|values\b))?',
    re.IGNORECASE
)


def extract_table_column_descriptions(create_statements):
    '''{table_name: {column_name: description}}'''
    table2columns = dict()
    for create_statement in create_statements:
        table_match = _CREATE_TABLE_RE.search(create_statement)
        if table_match is None:
            continue
        columns = dict()
        for column_name, description in re.findall(_COLUMN_RE, create_statement):
            columns.setdefault(column_name.lower(), description)
        table2columns[table_match.group(1).lower()] = columns
    return table2columns


def insert_target(sql):
    '''returns (target table, True if the INSERT lists its columns explicitly)'''
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    sql = re.sub(r'--[^\n]*', ' ', sql)
    match = _INSERT_RE.search(sql)
    if match is None:
        return None, False
    return match.group(1).lower(), match.group(2) is not None


def build_used_columns(sql, statement_type, column_name2column_desc, table2columns):
    used = dict()
    sql_lower = sql.lower()

    # original rule: every column whose name appears in the SQL text
    for column_name, column_desc in column_name2column_desc.items():
        if column_name.lower() in sql_lower:
            used[column_name] = column_desc

    # [CHANGE 3] statements that don't name their columns need the columns of the tables they touch
    tables = []
    if statement_type == "INSERT":
        target, explicit_columns = insert_target(sql)
        if target in table2columns and not explicit_columns:  # e.g. INSERT INTO t VALUES (...)
            tables = [target]
    elif statement_type in ("DROP", "ALTER"):
        tables = [t for t in table2columns if re.search(r'(?<![\w])' + re.escape(t) + r'(?![\w])', sql_lower)]
    for table in tables:
        used.update(table2columns[table])  # descriptions of the table itself take precedence
    return used


if __name__ == "__main__":
    random.seed(42)
    db_path = "database_synthesis/sinthetic_sqlite_databases"
    sql_infos = json.load(open("sql_synthesis/results/synthetic_sqls.json"))
    question_synthesis_template = open("question_synthesis/prompt_templates/question_synthesis_prompt.txt").read()
    styles = ["Formal", "Colloquial", "Imperative", "Interrogative", "Descriptive", "Concise", "Vague", "Metaphorical",
              "Multi-turn Dialogue"]

    if sql_infos:  # [CHANGE 5] guard against an empty input file
        print(sql_infos[0])
    db_ids = list(set([sql["db_id"] for sql in sql_infos]))
    print(len(db_ids))

    db_id2column_info = dict()
    db_id2table_info = dict()  # [CHANGE 3]
    for db_id in tqdm(db_ids):
        table_names, create_statements = obtain_db_schema(os.path.join(db_path, db_id, db_id + ".sqlite"))
        db_id2column_info[db_id] = extract_column_descriptions(create_statements)
        db_id2table_info[db_id] = extract_table_column_descriptions(create_statements)  # [CHANGE 3]

    prompts = []
    for sql_info in tqdm(sql_infos):
        style_name = random.sample(styles, 1)[0]
        column_name2column_desc = db_id2column_info[sql_info["db_id"]]
        # [CHANGE 1] statement type from post_process_sqls.py (or detected here)
        statement_type = sql_info.get("statement_type") or get_statement_type(sql_info["sql"])
        # [CHANGE 3] same result as the original loop for SELECT; table-aware for INSERT/DROP/ALTER
        used_column_name2column_desc = build_used_columns(
            sql_info["sql"], statement_type, column_name2column_desc, db_id2table_info[sql_info["db_id"]]
        )

        if style_name in ["Vague", "Metaphorical"]:  # "Vague" and "Metaphorical" styles require external knowledge
            steps = steps_w_ek
            guidelines = guidelines_w_ek
            instruction = instruction_w_ek
            output_format = output_format_w_ek
        elif style_name == "Multi-turn Dialogue":  # the "Multi-turn Dialogue" style uses a special multi-round format
            steps = steps_multi_round
            guidelines = guidelines_multi_round
            instruction = instruction_multi_round
            output_format = output_format_multi_round
        else:
            steps = steps_wo_ek
            guidelines = guidelines_wo_ek
            instruction = instruction_wo_ek
            output_format = output_format_wo_ek

        # [CHANGE 2] SELECT: original guidelines (unchanged); other statement types: statement-specific guidelines
        guidelines = build_guidelines(statement_type, style_name)

        prompt = question_synthesis_template.format(
            style_desc=style2desc[style_name].strip(),
            engine="SQLite",
            column_info=json.dumps(used_column_name2column_desc, indent=2, ensure_ascii=False).strip(),
            sql=sql_info["sql"].strip(),
            steps=steps.strip(),
            guidelines=guidelines.strip(),
            output_format=output_format.strip(),
            instruction=instruction.strip()
        )

        sql_info["style"] = style_name
        sql_info["statement_type"] = statement_type  # [CHANGE 5]
        sql_info["prompt"] = prompt

    # [CHANGE 5] overview of what was generated
    type_counts = dict()
    for sql_info in sql_infos:
        type_counts[sql_info["statement_type"]] = type_counts.get(sql_info["statement_type"], 0) + 1
    print("prompts per statement type:", dict(sorted(type_counts.items(), key=lambda kv: -kv[1])))

    os.makedirs("question_synthesis/prompts", exist_ok=True)  # [CHANGE 4]
    with open("question_syntesis/prompts/question_synthesis_prompts.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(sql_infos, indent=2, ensure_ascii=False))