

# Stylized Natural Language Question Synthesis

This is the third step in data synthesis framework, dedicated to generating stylized natural language questions for synthetic SQL queries.

## Step 1: Question Generation

Generate stylized natural language questions.

1. Run `python3 generate_question_synthesis_prompts.py` to create prompts for question generation.
2. Execute `python3 synthesize_question.py` to generate questions for the synthesized SQL queries.  For each prompt (SQL query), we sample multiple responses (questions) with a temperature of `0.8`.
Additional arguments:
   - --model - defines AI model that would be used. By default is Qwen3.5-2B
   - --num_samples - Responses per prompt. Default: 1
   - --batch_size - Number of prompts processed at once. Larger batches are faster but need more GPU memory. Default: 1
   - --max_new_tokens - Maximum tokens generated per response. Default: 1024
   - --max_prompt_tokens - Truncates prompts to this many tokens.
   - --thinking - Enables AI’s thinking mode. It is slower, but the model reasons before answering.
   - --greedy - Switches to deterministic decoding (always the most likely token). It forces num_samples to 1, because identical samples would be pointless. 
   - --temperature - Overrides the randomness of generation. Higher values give more varied questions.
   - --top_p - Overrides nucleus sampling, which limits the choice to the most probable tokens whose cumulative probability is top_p.
   - --num_dbs - Uses only the prompts from the first N distinct databases in the input file
   - --limit - Processes only the first N prompts.
   - --no_4bit - Loads the model in fp16 instead of 4-bit quantization.
   - --input - Default: question_synthesis/prompts/question_synthesis_prompts.json
   - --output - Default: question_synthesis/results/question_synthesis.json

## Step 2: Post-Processing

1. Execute `python3 post_process_questions.py` to perform semantic consistency selection, ensuring the generated questions align closely with their corresponding SQL queries.
2. The final synthetic results will be saved to `./results/question_and_sql_pairs.json`.