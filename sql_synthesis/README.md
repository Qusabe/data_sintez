# First of all

That's an old README from the original source. Will be updated later



# Complexity-Aware SQL Query Generation

This is the second step in our data synthesis framework, focused on generating complexity-aware SQL queries based on synthetic databases.

## Step 1: SQL Query Generation

Generate SQL queries by leveraging database schemas, database values, query complexity, and SQLite-supported functions.

1. Execute `python3 generate_sql_synthesis_prompts.py` to create prompts for SQL query generation.
2. Run `python3 synthesize_sql.py` to generate SQL queries using LLMs.
Additional arguments:
   - --model - defines AI model that would be used. By default is Qwen3.5-2B
   - --batch_size - 
   - --max_new_tokens - 
   - --max_prompt_tokens -
   - --thinking
   - --greedy
   - --temperature
   - --top_p
   - --num_dbs
   - --limit
   - --no_4bit
   - --cpu_offload
   - --gpu_mem_gb
   - --cpu_mem_gb
   - --input
   - --output
## Step 2: Post-Processing

Refine the generated SQL queries to ensure quality and remove invalid or redundant queries:

1. Run `python3 post_process_sqls.py` to:
   - Discard non-SELECT queries.
   - Remove queries with syntax errors or execution timeouts.
   - Deduplicate queries based on their templates.

2. The final synthetic SQL queries will be saved in `./results/synthetic_sqls.json`.
