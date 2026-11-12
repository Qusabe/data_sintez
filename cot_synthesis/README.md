\
# Chain-of-Thought Solution Synthesis

This is the fourth step in data synthesis framework, focused on generating step-by-step chain-of-thought (CoT) solutions

## Step 1: Chain-of-Thought Generation

Create CoT solutions for each data sample.

1. Run `python3 generate_cot_synthesis_prompts.py` to prepare prompts for CoT generation.
2. Execute `python3 synthesize_cot.py` to generate CoT solutions for data samples. For each prompt, we sample multiple CoT solutions with a temperature of `0.8`.
    Additional arguments:
   - --model - defines AI model that would be used. By default is Qwen3.5-2B
   - --num_samples -- CoT responses per prompt. More samples = higher chance that one is correct, but much slower. Default: 1
   - --batch_size - Number of prompts processed at once. Larger batches are faster but need more GPU memory. Default: 1
   - --max_new_tokens - Max generated tokens per response. A CoT is long: raise it (e.g. 4096) if responses get cut off. Default: 2048
   - --max_prompt_tokens - Truncates prompts to this many tokens.
   - --thinking - Enables AI’s thinking mode. It is slower, but the model reasons before answering.
   - --greedy - Switches to deterministic decoding (always the most likely token). It forces num_samples to 1, because identical samples would be pointless. 
   - --temperature - Overrides the randomness of generation. Higher values give more varied questions.
   - --top_p - Overrides nucleus sampling, which limits the choice to the most probable tokens whose cumulative probability is top_p.
   - --num_dbs - Uses only the prompts from the first N distinct databases in the input file
   - --limit - Processes only the first N prompts.
   - --no_4bit - Loads the model in fp16 instead of 4-bit quantization.
   - --input - Default: cot_synthesis/prompts/cot_synthesis_prompts.json
   - --output - Default: cot_synthesis/results/cot_synthesis.json

## Step 2: Post-Processing

1. Run `python3 post_process_cot.py` to perform execution-based major voting, selecting the most reliable CoT solutions.
2. Save the final synthetic `<database, question, SQL query, CoT solution>` samples to `./results/synthetic_text2sql_dataset.json`.
