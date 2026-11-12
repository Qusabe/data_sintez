# Translation of the obtained dataset into the required language

This is the final step in data synthesis framework, focused on translating Natural Language Questions and CoT (chain-of-thought) in the dataset that had been obtained in the previous step.  
  
We are making a translation to enhance the quality of the new dataset for using it for fine-tuning LLM models. Because of that, LLM model potentially could perform tasks of NL2<LANG> with higher precision, because it had been fine-tuned with dataset in the target language.
  
Translation are being executed by a pfnet/plamo-3-translate-8b LLM translation model by default. Script supports translation into 52 different languages that are listed in the langs/supported_langs.txt.   


# 1. Available arguments
Script has the following additional arguments so you can manage the execution of the script:
- --input - input JSON file (list of objects).
- --output - output file name. By default: <input name>_<lang>.json
- --target-lang, -t - target language: code (de, fr, ja, ...) or English name
- --list-languages - show all supported target languages and exit
- --result-dir - folder for all results; created if missing. By default: dataset_translation/results
- --model - HF model id or local directory. By default is pfnet/plamo-3-translate-8b
- --batch-size - lines per batch. If you have a low memory, better keep its value low. By default is 32.
- --max-new-tokens - upper limit of generated tokens per line. By default is 2048
- --limit - translate only the first N elements. Could be useful for testing. 

# 2. Run the script
This is an example of bash command that would allow you to run this script. You can modify it with an arguments listed above.
```bash
python translate_dataset cot_synthesis/results/synthetic_text2sql_dataset.json -t ru --model pfnet/plamo-3-translate-8b
```