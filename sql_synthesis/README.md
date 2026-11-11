
# Complexity-Aware SQL Query Generation

This is the second step in  data synthesis framework, focused on generating complexity-aware SQL queries based on synthetic databases.

## Step 1: SQL Query Generation

Generate SQL queries by leveraging database schemas, database values, query complexity, and SQLite-supported functions.

1. Execute `python3 generate_sql_synthesis_prompts.py` to create prompts for SQL query generation. Additional arguments:
   - --db-path - Folder containing the databases, each as <db_name>/<db_name>,sqlite. Default: database_synthesis/sinthetic_sqlite_databases
   - --template - Text file with the prompt template. It is filled in with the schema, example rows, SQL functions, statement type, complexity criteria, and so on. Default: sql_synthesis/prompt_templates/sql_synthesis_prompt.txt
   - --funcs - JSON list of SQLite function descriptions. Prompts randomly include 0 to 2 of them as hints. Default: sql_synthesis/prompt_templates/sqlite_funcs.json 
   - --output - where the prompts are saved. The folder is created automatically. Default: sql_synthesis/prompts/sql_synthesis_prompts.json
   - --num-dbs -- Number of databases to use. They are chosen randomly (reproducibly, via the seed). Databases with no tables are skipped and do not count toward this number.
   - --prompts-per-db - Number of prompts generated for each databse. Default: 300
   - --statement-types - One or more statement types to sample from, for example --statement-types SELECT UPDATE. For each prompt one type is picked at random. Default: SELECT INSERT UPDATE DELETE
   - --seed - Seeds both random and numpy, so the same arguments produce the ssame databse selection and the same prompts. Default: 42
2. Run `python3 synthesize_sql.py` to generate SQL queries using LLMs.
Additional arguments:
   - --model - defines AI model that would be used. By default is Qwen3.5-2B
   - --batch_size - Number of prompts processed at once. Larger batches are faster but need more GPU memory. Default: 1
   - --max_new_tokens - Maximum tokens generated per response. Default: 4096
   - --max_prompt_tokens - Truncates prompts to this many tokens.
   - --thinking - Enables AI’s thinking mode. It is slower, but the model reasons before answering.
   - --greedy - Switches to deterministic decoding (always the most likely token). It forces num_samples to 1, because identical samples would be pointless. 
   - --temperature - Overrides the randomness of generation. Higher values give more varied questions.
   - --top_p - Overrides nucleus sampling, which limits the choice to the most probable tokens whose cumulative probability is top_p.
   - --num_dbs - Uses only the prompts from the first N distinct databases in the input file
   - --limit - Processes only the first N prompts.
   - --no_4bit - Loads the model in fp16 instead of 4-bit quantization.
   - --cpu_offload - Lets layers that don’t fit in GPU memory stay on the CPU (in fp32).
   - --gpu_mem_gb - Only used with --cpu_offload. Caps GPU memory use, for example 2.5 on a 4 GB card, which leaves room for activations and the KV cache. Everything above the cap goes to CPU RAM.
   - --cpu_mem_gb - Only used with --cpu_offload. Caps CPU RAM use for the offloaded parts. Default: 16
   - --input - JSON file with the prompts. Default: sql_synthesis/prompts/sql_synthesis_prompts.json
   - --output - Where results are saved. The folder is created automatically if it is missing. Default: sql_synthesis/results/sql_synthesis.json
## Step 2: Post-Processing

Refine the generated SQL queries to ensure quality and remove invalid or redundant queries:

1. Run `python3 post_process_sqls.py` to:
   - Discard non-SELECT queries.
   - Remove queries with syntax errors or execution timeouts.
   - Deduplicate queries based on their templates.

2. The final synthetic SQL queries will be saved in `./results/synthetic_sqls.json`.
