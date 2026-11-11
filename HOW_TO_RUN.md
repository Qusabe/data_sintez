# Pipeline 

The pipeline has three stages: SQL synthesis, question synthesis, and chain-of-thought (CoT) synthesis. Each stage generates prompts, runs the model, and post-processes the results.  
  
Parameters values that are shown has been used by me in my test runs on my local PC, so you should to change them accordingly to your tasks, desires and specs of your machine that would be running the code.

## Stage 1: SQL synthesis

### 1. Generate SQL synthesis prompts

```bash
python3 sql_synthesis/generate_sql_synthesis_prompts.py --num-dbs 5 --prompts-per-db 3
```

Total prompts = `--num-dbs` × `--prompts-per-db`.

With a sample of 750 databases and 20 prompts per database, this gives 15,000 prompts. Approximately 2/3 of them are discarded during the pipeline, so the final dataset should contain approximately 5,000 elements.  And such a command would look
the following way:

```bash
python3 sql_synthesis/generate_sql_synthesis_prompts.py --num-dbs 750 --prompts-per-db 20
```


### 2. Synthesize SQL

```bash
python3 sql_synthesis/synthesize_sql.py --model Qwen/Qwen3.5-2B --batch_size 1 --max_new_tokens 1024
```

These arguments were used in the test runs on a local PC. On more powerful machines, or with models that have more parameters, adjust them to your hardware for faster and better results.

Example:

```bash
python3 sql_synthesis/synthesize_sql.py --model Qwen/Qwen3.5-9B --batch_size 3 --max_new_tokens 2048
```

### 3. Post-process SQL

```bash
python3 sql_synthesis/post_process_sqls.py
```

## Stage 2: Question synthesis

### 4. Generate question synthesis prompts

```bash
python3 question_synthesis/generate_question_synthesis_prompts.py
```

### 5. Synthesize questions

```bash
python3 question_synthesis/synthesize_question.py
```

### 6. Post-process questions

```bash
python3 question_synthesis/post_process_questions.py
```

## Stage 3: CoT synthesis

### 7. Generate CoT synthesis prompts

```bash
python3 cot_synthesis/generate_cot_synthesis_prompts.py
```

### 8. Synthesize CoT

```bash
python3 cot_synthesis/synthesize_cot.py
```

### 9. Post-process CoT

```bash
python3 cot_synthesis/post_process_cot.py
```

