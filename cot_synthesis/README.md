# First of all

That's an old README from the original source. Will be updated later


# Chain-of-Thought Solution Synthesis

This is the final step in data synthesis framework, focused on generating step-by-step chain-of-thought (CoT) solutions

## Step 1: Chain-of-Thought Generation

Create CoT solutions for each data sample.

1. Run `python3 generate_cot_synthesis_prompts.py` to prepare prompts for CoT generation.
2. Execute `python3 synthesize_cot.py` to generate CoT solutions for data samples. For each prompt, we sample multiple CoT solutions with a temperature of `0.8`.
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
   - --input
   - --output

## Step 2: Post-Processing

1. Run `python3 post_process_cot.py` to perform execution-based major voting, selecting the most reliable CoT solutions.
2. Save the final synthetic `<database, question, SQL query, CoT solution>` samples to `./results/synthetic_text2sql_dataset.json`.
