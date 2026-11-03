# import argparse
# import json
# import os
# import re
# import time
# import json_repair
#
# def parse_response(response):
#     domain_pattern = r'(?<=\[START_DOMAIN\])(.*?)(?=\[END_DOMAIN\])'
#     scenario_pattern = r'(?<=\[START_SCENARIO\])(.*?)(?=\[END_SCENARIO\])'
#     schema_pattern = r'(?<=\[START_DATABASE_SCHEMA\])(.*?)(?=\[END_DATABASE_SCHEMA\])'
#
#     try:
#         domain_match = re.search(domain_pattern, response, re.DOTALL)
#         domain = domain_match.group(0).strip() if domain_match else None
#
#         scenario_match = re.search(scenario_pattern, response, re.DOTALL)
#         scenario = scenario_match.group(0).strip() if scenario_match else None
#
#         schema_match = re.search(schema_pattern, response, re.DOTALL)
#         schema = schema_match.group(0).strip() if schema_match else None
#         schema_dict = json_repair.loads(schema)
#         schema = json.dumps(schema_dict, indent=2, ensure_ascii=False)
#
#         return domain, scenario, schema
#     except Exception as e:
#         print(response)
#         print("Parsing Exception:", str(e))
#         return None, None, None
#
# def llm_inference(model, prompts):
#     '''
#     This function leverages a large language model (LLM) to generate responses for a given list of prompts.
#     You can integrate your preferred LLM within this function.
#
#     Args:
#         model: The LLM to be used for inference.
#         prompts: A list of prompts for which the LLM will generate responses.
#
#     Returns:
#         A list of dictionaries containing the prompt, the generated response, and extracted components
#         (domain, scenario, schema) from the response. Invalid responses are filtered out.
#     '''
#
#     # Generate responses using the LLM (each prompt corresponds to one response)
#     responses = None  # Replace this with the actual LLM call, e.g., model.generate(prompts, temperature=0, n=1)
#
#     # Initialize a list to store the processed results
#     results = []
#
#     # Iterate over prompts and their corresponding responses
#     for prompt, response in zip(prompts, responses):
#         # Parse the response to extract domain, scenario, and schema
#         domain, scenario, schema = parse_response(response)
#
#         # Filter out invalid responses where any component is missing
#         if domain is None or scenario is None or schema is None:
#             continue
#
#         # Append valid results to the list
#         results.append({
#             "prompt": prompt,
#             "generated_content": {
#                 "response": response,
#                 "domain": domain,
#                 "scenario": scenario,
#                 "schema": schema
#             }
#         })
#
#     return results
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type = str)
#     args = parser.parse_args()
#
#     print(args)
#
#     prompts = json.load(open("./prompts/prompts_schema_synthesis.json"))
#     output_file = "./results/schema_synthesis.json"
#     results = llm_inference(args.model, prompts)
#
#     with open(output_file, "w", encoding = "utf-8") as f:
#         f.write(json.dumps(results, indent = 2, ensure_ascii = False))

import os

# Must be set before torch is imported to reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re

import json_repair
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_response(response):
    domain_pattern = r'(?<=\[START_DOMAIN\])(.*?)(?=\[END_DOMAIN\])'
    scenario_pattern = r'(?<=\[START_SCENARIO\])(.*?)(?=\[END_SCENARIO\])'
    schema_pattern = r'(?<=\[START_DATABASE_SCHEMA\])(.*?)(?=\[END_DATABASE_SCHEMA\])'

    try:
        domain_match = re.search(domain_pattern, response, re.DOTALL)
        domain = domain_match.group(0).strip() if domain_match else None

        scenario_match = re.search(scenario_pattern, response, re.DOTALL)
        scenario = scenario_match.group(0).strip() if scenario_match else None

        schema_match = re.search(schema_pattern, response, re.DOTALL)
        schema = schema_match.group(0).strip() if schema_match else None
        schema_dict = json_repair.loads(schema)
        schema = json.dumps(schema_dict, indent=2, ensure_ascii=False)

        return domain, scenario, schema
    except Exception as e:
        print(response)
        print("Parsing Exception:", str(e))
        return None, None, None


def load_model(model_name, use_4bit=True):
    """Load a Hugging Face causal LM and its tokenizer on the GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU found. In Colab: Runtime -> Change runtime type -> T4 GPU, then reconnect."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    kwargs = {"device_map": "auto"}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["dtype"] = torch.float16  # T4 has no good bf16 support

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def generate_responses(model, tokenizer, prompts, batch_size=1, max_new_tokens=2048,
                       max_prompt_tokens=None):
    """
    Generate one response per prompt with greedy decoding (temperature=0 equivalent).

    - Prompts are sorted by length so each batch has little padding (less memory).
    - If a batch runs out of GPU memory, its prompts are retried one at a time.
    - A prompt that still doesn't fit is skipped (empty response, filtered out later).
    """
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # ignored by templates that don't use it (e.g. Llama)
        )
        for p in prompts
    ]

    lengths = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
    print(f"Prompt length in tokens: max={max(lengths)}, avg={sum(lengths) / len(lengths):.0f}")

    tok_kwargs = {}
    if max_prompt_tokens:
        tok_kwargs = {"truncation": True, "max_length": max_prompt_tokens}

    order = sorted(range(len(texts)), key=lambda i: lengths[i])
    responses = [None] * len(texts)

    def run(batch_idx):
        batch = [texts[i] for i in batch_idx]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, add_special_tokens=False, **tok_kwargs
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Drop the prompt tokens and decode only what the model generated
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, r in zip(batch_idx, decoded):
            responses[i] = r

    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]
        try:
            run(batch_idx)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            for i in batch_idx:  # retry one by one
                try:
                    run([i])
                except torch.OutOfMemoryError:
                    print(f"Skipping prompt {i} ({lengths[i]} tokens): too long for GPU memory")
                    responses[i] = ""
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        print(f"Generated {min(start + batch_size, len(order))}/{len(order)}")

    return responses


def llm_inference(model, prompts, batch_size=1, max_new_tokens=2048, use_4bit=True,
                  max_prompt_tokens=None):
    '''
    This function leverages a large language model (LLM) to generate responses for a given list of prompts.

    Args:
        model: Hugging Face model name or local path (e.g. Qwen/Qwen3-4B or meta-llama/Llama-3.2-3B-Instruct).
        prompts: A list of prompts for which the LLM will generate responses.

    Returns:
        A list of dictionaries containing the prompt, the generated response, and extracted components
        (domain, scenario, schema) from the response. Invalid responses are filtered out.
    '''

    llm, tokenizer = load_model(model, use_4bit=use_4bit)

    # Generate responses using the LLM (each prompt corresponds to one response)
    responses = generate_responses(
        llm, tokenizer, prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )

    # Initialize a list to store the processed results
    results = []

    # Iterate over prompts and their corresponding responses
    for prompt, response in zip(prompts, responses):
        # Skip prompts that were too long to process
        if not response:
            continue

        # Parse the response to extract domain, scenario, and schema
        domain, scenario, schema = parse_response(response)

        # Filter out invalid responses where any component is missing
        if domain is None or scenario is None or schema is None:
            continue

        # Append valid results to the list
        results.append({
            "prompt": prompt,
            "generated_content": {
                "response": response,
                "domain": domain,
                "scenario": scenario,
                "schema": schema
            }
        })

    print(f"Kept {len(results)}/{len(prompts)} valid responses")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_prompt_tokens", type=int, default=None,
                        help="Truncate prompts to this many tokens (cuts off the END of the prompt)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N prompts (quick test run)")
    parser.add_argument("--no_4bit", action="store_true", help="Load in fp16 instead of 4-bit")
    args = parser.parse_args()

    print(args)

    with open("database_synthesis/prompts/prompts_schema_synthesis.json", encoding="utf-8") as f:
        prompts = json.load(f)
    if args.limit:
        prompts = prompts[:args.limit]

    output_file = "database_synthesis/results/schema_synthesis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = llm_inference(
        args.model,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_4bit=not args.no_4bit,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} results to {output_file}")