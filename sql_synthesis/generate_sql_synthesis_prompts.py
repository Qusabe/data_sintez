import argparse
import json
import os
import random
import sqlite3
import numpy as np

from tqdm import tqdm

sql_func_template = '''
### SQL Functions
You may consider one or more of the following SQL functions while generating the query:
{sql_funcs}
Important tips:
Except for the functions listed above, you may use any other functions as long as they conform to the syntax of the database engine.
'''

insert_stmts_template = '''
### INSERT INTO Statements
Below are several `INSERT INTO` statements. Use these to help generate predicates (i.e., `WHERE` clauses) in your SQL query:

{insert_statements}
'''

simple_criterion = '''**Criteria:**
Simple SQL queries may satisfy one or more of the following criteria:
- Simple queries should select data from a single table only.
- Basic aggregate functions are permitted, such as `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
- No joins are allowed; the query must operate on a single table.

**Example of Simple SQL Query:**
```sql
SELECT name, department_name
FROM employees
WHERE level > 5
ORDER BY age DESC;
```'''

moderate_criterion = '''**Criteria:**
Moderate SQL queries may satisfy one or more of the following criteria:
- Involves table joins, such as `JOIN`, `INNER JOIN`, `LEFT JOIN`, `CROSS JOIN`, etc.
- Includes subqueries within the `SELECT` or `WHERE` clauses.
- Utilizes aggregate functions alongside a `GROUP BY` clause.
- Contains complex `WHERE` conditions, including `IN`, `BETWEEN`, `LIKE`.
- Incorporate a `HAVING` clause to filter aggregated results.
- Uses aggregate functions like `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, etc.

**Example of Moderate SQL Query:**
```sql
SELECT e.name, d.department_name, AVG(s.salary) AS average_salary
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
LEFT JOIN salaries s ON e.employee_id = s.employee_id
WHERE e.age > 30 AND e.status = 'active'
GROUP BY e.name, d.department_name
HAVING AVG(s.salary) > 50000;
```'''

complex_criterion = '''**Criteria:**
Complex SQL queries may satisfy one or more of the following criteria:
- Contains complex nested subqueries.
- Utilizes multiple types of joins, including self-joins.
- Includes window functions, such as `ROW_NUMBER`, `RANK`, etc.
- Uses Common Table Expressions (CTEs) for improved readability.
- Combines multiple aggregate functions.
- Involves complex `WHERE` and `HAVING` clauses with multiple conditions.
- Utilizes advanced functions and operators.

**Example of Complex SQL Query:**
```sql
WITH EmployeeCTE AS (
    SELECT employee_id, name, department_id, ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT e.name, d.department_name
FROM EmployeeCTE e
INNER JOIN departments d ON e.department_id = d.department_id
WHERE e.rank <= 3;
```'''

highly_complex_criterion = '''**Criteria:**
Highly complex SQL queries may satisfy one or more of the following criteria:
- Includes multiple Common Table Expressions (CTEs) for readability.
- Combines nested subqueries and various joins.
- Utilizes recursive CTEs for hierarchical or recursive queries.
- Extensively uses advanced window functions.
- May involve `UNION` or `UNION ALL` to combine result sets.
- Implements complex logic with advanced analytical functions.
- Employs a wide range of SQL clauses and conditions.
- Utilizes a broad spectrum of SQL functions and advanced features.

**Example of Highly Complex SQL Query:**
```sql
WITH RECURSIVE EmployeeHierarchy AS (
    SELECT employee_id, name, manager_id, department_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.employee_id, e.name, e.manager_id, e.department_id, eh.level + 1
    FROM employees e
    JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
),
DepartmentSalaries AS (
    SELECT eh.employee_id, eh.name, eh.level, d.department_name, s.salary, d.department_id
    FROM EmployeeHierarchy eh
    INNER JOIN departments d ON eh.department_id = d.department_id
    INNER JOIN salaries s ON eh.employee_id = s.employee_id
),
DepartmentStats AS (
    SELECT 
        d.department_id,
        COUNT(e.employee_id) AS employee_count,
        AVG(s.salary) AS average_salary
    FROM employees e
    INNER JOIN salaries s ON e.employee_id = s.employee_id
    INNER JOIN departments d ON e.department_id = d.department_id
    GROUP BY d.department_id
)
SELECT ds.name, ds.level, 
    SUM(ds.salary) OVER (PARTITION BY ds.department_id ORDER BY ds.level, ds.name) AS cumulative_salary
FROM DepartmentSalaries ds
INNER JOIN DepartmentStats dstat ON ds.department_id = dstat.department_id
ORDER BY ds.level, ds.name;
```'''

# Statement types. The text values are my own wording - edit freely.
# The complexity criteria above are written with SELECT queries in mind, so for the
# data-modifying types a short note tells the model how to apply them.
non_select_complexity_note = (
    "Apply the complexity criteria below to this statement type wherever they make sense "
    "(e.g. subqueries, CTEs, joins expressed via `UPDATE ... FROM` or subqueries, "
    "`INSERT ... SELECT`). Where a criterion cannot apply to this statement type, "
    "satisfy the closest equivalent."
)

statement_type2desc = {
    "SELECT": "Retrieve data from the database without modifying it.",
    "INSERT": (
        "Add new rows to a table, using explicit `VALUES` or `INSERT ... SELECT`. "
        "The statement must be valid for the schema: respect NOT NULL, PRIMARY KEY, "
        "UNIQUE and FOREIGN KEY constraints.\n" + non_select_complexity_note
    ),
    "UPDATE": (
        "Modify values in existing rows of a table. "
        "Use a meaningful `WHERE` clause so that only the intended rows are changed.\n"
        + non_select_complexity_note
    ),
    "DELETE": (
        "Remove existing rows from a table. "
        "Use a meaningful `WHERE` clause so that only the intended rows are removed.\n"
        + non_select_complexity_note
    ),
}


# The old script always asked for "column_count" columns, which only makes
# sense for SELECT. This builds a requirement sentence appropriate for each type.
def build_column_requirement(statement_type, column_count):
    if statement_type == "SELECT":
        return f"The query must return exactly {column_count} column(s)."
    if statement_type == "INSERT":
        return ("Explicitly list the target columns in the `INSERT` statement and supply "
                "valid values for all columns that require them.")
    if statement_type == "UPDATE":
        return f"The `SET` clause must modify exactly {column_count} column(s)."
    if statement_type == "DELETE":
        return "The statement must contain a `WHERE` clause so that it does not delete every row."
    raise ValueError(f"Unknown statement type: {statement_type}")


def obtain_db_schema(db_file_dir):
    conn = sqlite3.connect(db_file_dir)
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


def obtain_insert_statements(db_file_dir, table_names):
    table_name2insert_statements = dict()
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    for table_name in table_names:
        try:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 2')
            rows = cursor.fetchall()

            column_names = [description[0] for description in cursor.description]

            insert_statements = []
            for row in rows:
                values = ', '.join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in row])
                insert_statement = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({values});"
                insert_statements.append(insert_statement)

            # for statement in insert_statements:
            #     print(statement)
            table_name2insert_statements[table_name] = insert_statements

        except Exception as e:
            print(e)

    cursor.close()
    conn.close()

    return table_name2insert_statements


#  Command-line arguments replace the hard-coded paths/values.
def parse_args():
    p = argparse.ArgumentParser(description="Generate SQL-synthesis prompts from SQLite databases.")
    p.add_argument("--db-path", default="database_synthesis/sinthetic_sqlite_databases",
                   help="Folder containing <db_name>/<db_name>.sqlite")
    p.add_argument("--template", default="sql_synthesis/prompt_templates/sql_synthesis_prompt.txt")
    p.add_argument("--funcs", default="sql_synthesis/prompt_templates/sqlite_funcs.json")
    p.add_argument("--output", default="sql_synthesis/prompts/sql_synthesis_prompts.json")
    # how many databases to use (None = all of them)
    p.add_argument("--num-dbs", type=int, default=None,
                   help="Number of databases to use (randomly chosen). Default: all.")
    # how many prompts per database (original value was 300)
    p.add_argument("--prompts-per-db", type=int, default=300,
                   help="Number of prompts generated for each database. Default: 300.")
    # which statement types to generate
    p.add_argument("--statement-types", nargs="+", default=list(statement_type2desc.keys()),
                   choices=list(statement_type2desc.keys()),
                   help="Statement types to sample from. Default: SELECT INSERT UPDATE DELETE.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)  #  the original seeded only `random`, not numpy

    db_path = "database_synthesis/sinthetic_sqlite_databases"
    prompt_template = open(args.template, "r", encoding="utf-8").read()
    functions = json.load(open(args.funcs))

    complexity2criterion = {
        "Simple": simple_criterion,
        "Moderate": moderate_criterion,
        "Complex": complex_criterion,
        "Highly Complex": highly_complex_criterion
    }

    #  Shuffle (reproducibly) and walk through the databases until `num_dbs`
    # usable ones have been processed. Sorting first makes the result independent of
    # the order in which the OS lists the folders.
    db_names = sorted(os.listdir(db_path))
    random.shuffle(db_names)
    target_dbs = args.num_dbs if args.num_dbs is not None else len(db_names)

    prompts = []
    used_dbs = 0
    progress = tqdm(total=min(target_dbs, len(db_names)), desc="Databases")
    for db_name in db_names:
        if used_dbs >= target_dbs:
            break
        try:
            db_file_dir = os.path.join(db_path, db_name, db_name + ".sqlite")
            table_names, create_statements = obtain_db_schema(db_file_dir)

            # skip databases that contain no tables (they would give empty schemas)
            if len(table_names) == 0:
                continue

            table_name2insert_statements = obtain_insert_statements(db_file_dir, table_names)

            for _ in range(0, args.prompts_per_db):  #  was: range(0, 300)
                complexity = random.sample(["Simple", "Moderate", "Complex", "Highly Complex"], 1)[0]

                #  pick the statement type (was implicitly always SELECT)
                statement_type = random.choice(args.statement_types)

                insert_statements = []
                for table_name in table_names:
                    insert_statements += table_name2insert_statements.get(table_name, [])

                if len(insert_statements) == 0:
                    db_value_prompt = ""
                else:
                    if len(insert_statements) > 4:
                        insert_statements = random.sample(insert_statements, 4)
                    db_value_prompt = insert_stmts_template.format(insert_statements="\n\n".join(insert_statements))

                function_num = random.randint(0, 2)
                if function_num == 0:
                    sql_function_prompt = "### SQL Functions\nYou can use any function supported by the database engine."
                else:
                    sql_funcs = ""
                    sampled_functions = random.sample(functions, function_num)
                    for idx, func in enumerate(sampled_functions):
                        sql_funcs += f"Function {idx + 1}:\n" + func.strip() + "\n"
                    sql_function_prompt = sql_func_template.format(sql_funcs=sql_funcs)

                column_count = np.random.geometric(0.6, 1)[0]
                prompt = prompt_template.format(
                    schema_str="\n\n".join(create_statements),
                    sql_function_prompt=sql_function_prompt.strip(),
                    db_value_prompt=db_value_prompt.strip(),
                    statement_type=statement_type,                                    # [CHANGE 4]
                    statement_type_desc=statement_type2desc[statement_type],          # [CHANGE 4]
                    complexity=complexity,
                    criterion=complexity2criterion[complexity].strip(),
                    db_engine="SQLite",
                    column_requirement=build_column_requirement(statement_type, column_count)  # [CHANGE 4]
                )

                #  store statement_type and complexity next to the prompt
                prompts.append({
                    "prompt": prompt,
                    "db_id": db_name,
                    "statement_type": statement_type,
                    "complexity": complexity,
                })

            used_dbs += 1
            progress.update(1)
        except Exception as e:
            print(f"[{db_name}] {e}")
    progress.close()

    print(f"Used {used_dbs} databases, generated {len(prompts)} prompts.")

    #  create the output folder if it doesn't exist yet
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(prompts, indent=2, ensure_ascii=False))