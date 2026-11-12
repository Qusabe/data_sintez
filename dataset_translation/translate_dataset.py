
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tqdm import tqdm  # optional progress bar
except ImportError:  # pragma: no cover
    tqdm = None


# Settings
FIELDS_TO_TRANSLATE = ("question", "cot")  # keys that get translated
SOURCE_LANG = "en"                          # the dataset is in English
DEFAULT_TARGET_LANG = "ru"                  # used when --target-lang is omitted
DEFAULT_MODEL = "pfnet/plamo-3-translate-8b"  # 8B model
DEFAULT_RESULTS_DIR = "dataset_translation/results"                  # created automatically


SUPPORTED_LANGUAGES = {
    "ja": "Japanese", "en": "English", "zh-hans": "Chinese (Simplified)",
    "zh-hant": "Chinese (Traditional)", "ko": "Korean", "ar": "Arabic",
    "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "ceb": "Cebuano", "de": "German", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "it": "Italian",
    "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "lt": "Lithuanian", "lv": "Latvian", "ml": "Malayalam", "mr": "Marathi",
    "ms": "Malay", "my": "Burmese", "ne": "Nepali", "nl": "Dutch",
    "pa": "Punjabi", "pl": "Polish", "ru": "Russian", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "sr": "Serbian", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Tagalog",
    "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
}

# a few common aliases that users are likely to type.
LANGUAGE_ALIASES = {"zh": "zh-hans", "zh-cn": "zh-hans", "zh-tw": "zh-hant"}


def resolve_language(value: str) -> str:
    """
    Turn user input ("de", "German", "zh-cn", ...) into a valid model
    language code. Exits with a message if the language is unknown.
    """
    key = value.strip().lower()
    key = LANGUAGE_ALIASES.get(key, key)

    if key in SUPPORTED_LANGUAGES:                       # a language code
        code = key
    else:                                                # an English name
        by_name = {name.lower(): c for c, name in SUPPORTED_LANGUAGES.items()}
        if key not in by_name:
            sys.exit(f"Unsupported language: '{value}'. "
                     f"Run with --list-languages to see the supported ones.")
        code = by_name[key]

    if code == SOURCE_LANG:
        sys.exit("The target language is the same as the source language (English).")
    return code


def print_languages():
    """Print the table of supported target languages."""
    print("Supported target languages (use the code or the name):")
    for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda kv: kv[1]):
        if code != SOURCE_LANG:
            print(f"  {code:<8} {name}")


# Tokens the model may emit at the end of a translation; cut everything after.
LEFTOVER_TAG_RE = re.compile(r"<\|plamo:[a-z]+\|>.*$", re.S)



# Markdown / code protection

# A line that opens or closes a fenced code block.
FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Leading Markdown structure we do NOT want the model to see:
# indentation, bullets (* + -), numbered items (1. / 1)), headings (#), quotes (>)
PREFIX_RE = re.compile(r"^(\s*(?:(?:#{1,6}|[*+\-]|\d+[.)]|>)[ \t]+)*)(.*)$", re.S)

# Inline code: `code` or ``code``
INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")

# Placeholder that replaces inline code during translation. The regex is
# tolerant to spaces the model might add, e.g. "[[ C0 ]]".
PLACEHOLDER_RE = re.compile(r"\[\[\s*C\s*(\d+)\s*\]\]")


@dataclass
class Segment:
    """One translatable line of text (without its Markdown prefix)."""
    prefix: str                 # indentation / bullet, re-attached unchanged
    original: str               # original text of the line (with inline code)
    masked: str                 # text with inline code replaced by [[Cn]]
    codes: list = field(default_factory=list)  # the original inline code spans
    suffix: str = ""            # trailing whitespace, re-attached unchanged


def has_translatable_text(masked: str) -> bool:
    """True if the line contains at least one Latin letter outside placeholders."""
    return re.search(r"[A-Za-z]", PLACEHOLDER_RE.sub("", masked)) is not None


def parse_text(text: str) -> list:
    """
    Split a field value into a list of items. Each item is either
      * a plain string  -> kept verbatim (code blocks, blank lines, rules, ...)
      * a Segment       -> a line that must be translated
    Joining the items back with newlines reproduces the original layout.
    """
    items = []
    in_fence = False

    for line in text.split("\n"):
        # Fenced code block boundaries and everything inside them stay as is.
        if FENCE_RE.match(line):
            in_fence = not in_fence
            items.append(line)
            continue
        if in_fence:
            items.append(line)
            continue

        # Separate Markdown prefix from the text and trailing whitespace.
        prefix, rest = PREFIX_RE.match(line).groups()
        content = rest.rstrip()
        suffix = rest[len(content):]

        # Mask inline code with placeholders.
        codes = []

        def mask(match):
            codes.append(match.group(0))      # keep the code with its backticks
            return f"[[C{len(codes) - 1}]]"

        masked = INLINE_CODE_RE.sub(mask, content)

        if not has_translatable_text(masked):
            items.append(line)                # nothing to translate
        else:
            items.append(Segment(prefix, content, masked, codes, suffix))

    return items


def restore_code(translated: str, codes: list):
    """
    Put the original inline code back in place of the placeholders.
    Returns None if the model lost, duplicated or invented a placeholder.
    """
    found = [int(m.group(1)) for m in PLACEHOLDER_RE.finditer(translated)]
    if sorted(found) != list(range(len(codes))):
        return None
    return PLACEHOLDER_RE.sub(lambda m: codes[int(m.group(1))], translated)


def clean_output(text: str) -> str:
    """Remove special tags and line breaks that the model may leave behind."""
    text = LEFTOVER_TAG_RE.sub("", text)
    return " ".join(text.split("\n")).strip()



# Translator
class HFTranslator:
    """Runs the model locally Transformers."""
    def __init__(self, model_id: str, target_lang: str, max_new_tokens_cap: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.target_lang = target_lang
        self.cap = max_new_tokens_cap

        print(f"Loading model {model_id} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"   # required for batched generation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, dtype="bfloat16", device_map="auto"
            )
        except TypeError:  # older transformers versions use torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
            )
        self.model.eval()

    def translate_batch(self, texts: list) -> list:
        # The chat template builds the PLaMo prompt (PFML3 format)
        messages = [[{"role": "user", "content": t}] for t in texts]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            source_lang=SOURCE_LANG,
            target_lang=self.target_lang,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]
        #Some languages need more tokens
        max_new = min(self.cap, int(input_len * 4) + 64)

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new)

        return self.tokenizer.batch_decode(output[:, input_len:], skip_special_tokens=True)



# Translation driver with caching (resumable)
def translate_missing(texts, cache: dict, backend, batch_size: int, label: str):
    todo = sorted({t for t in texts if t not in cache}, key=len, reverse=True)
    if not todo:
        return

    batches = range(0, len(todo), batch_size)
    iterator = tqdm(batches, desc=label, unit="batch") if tqdm else batches

    for start in iterator:
        batch = todo[start:start + batch_size]
        for source, result in zip(batch, backend.translate_batch(batch)):
            cache[source] = clean_output(result)

        if not tqdm:
            print(f"{label}: {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)


def build_text(items: list, cache: dict, stats: dict) -> str:
    """Assemble the translated field value from parsed items and the cache."""
    lines = []
    for item in items:
        if isinstance(item, str):          # verbatim line
            lines.append(item)
            continue

        # 1st choice: translation of the masked line with code restored.
        translated = cache.get(item.masked, "")
        restored = restore_code(translated, item.codes) if translated else None

        # 2nd choice: translation of the unmasked line (inline code left in text).
        if restored is None and item.codes:
            fallback = cache.get(item.original, "")
            restored = fallback or None
            if restored:
                stats["fallback"] += 1

        # Last resort: keep the English original.
        if not restored:
            restored = item.original
            stats["kept_english"] += 1

        lines.append(f"{item.prefix}{restored}{item.suffix}")
    return "\n".join(lines)




def main():
    parser = argparse.ArgumentParser(
        description="Translate question/cot fields from English into a chosen language "
                    "with PLaMo 3 Translate 8B.")
    parser.add_argument("input", nargs="?", help="input JSON file (list of objects)")
    parser.add_argument("output", nargs="?", default=None,
                        help="output file name (default: <input name>_<lang>.json); "
                             "it is always created inside the results folder")
    # NEW: target language chosen by the user.
    parser.add_argument("--target-lang", "-t", default=DEFAULT_TARGET_LANG,
                        help="target language: code (de, fr, ja, ...) or English name "
                             f"(German, French, ...). Default: {DEFAULT_TARGET_LANG}")
    # NEW: print supported languages and exit.
    parser.add_argument("--list-languages", action="store_true",
                        help="show all supported target languages and exit")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="folder for all results; created if missing (default: results)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HF model id or local directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="lines per batch (lower it if you run out of memory)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="upper limit of generated tokens per line")
    parser.add_argument("--limit", type=int, default=None,
                        help="translate only the first N elements (for testing)")
    args = parser.parse_args()


    if args.list_languages:
        print_languages()
        return
    if not args.input:
        parser.error("the input file is required")
    target_lang = resolve_language(args.target_lang)
    print(f"Target language: {SUPPORTED_LANGUAGES[target_lang]} ({target_lang})")

    # ---- Prepare folders and file paths ---------------------------------------
    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f"Input file not found: {in_path}")

    # Create the results folder if it does not exist.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # The output file lives inside the results folder.
    out_name = Path(args.output).name if args.output else f"{in_path.stem}_{target_lang}.json"
    out_path = results_dir / out_name


    # Never overwrite the original input file.
    if out_path.resolve() == in_path.resolve():
        sys.exit("The output path would overwrite the input file. Choose another name.")

    #Load data
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        sys.exit("Expected the input JSON to be a list of objects.")
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} elements.")

    #Split every field into translatable lines
    parsed = []  # (element index, key, items)
    masked_texts = []
    for idx, element in enumerate(data):
        for key in FIELDS_TO_TRANSLATE:
            value = element.get(key)
            if isinstance(value, str) and value.strip():
                items = parse_text(value)
                parsed.append((idx, key, items))
                masked_texts.extend(i.masked for i in items if isinstance(i, Segment))

    unique = len(set(masked_texts))
    print(f"{len(masked_texts)} lines to translate ({unique} unique).")

    #Create the translator
    backend = HFTranslator(args.model, target_lang, args.max_new_tokens)

    #In-memory cache
    cache = {}

    #Pass 1: translate lines with inline code masked
    translate_missing(masked_texts, cache, backend, args.batch_size, "Pass 1")

    #Pass 2: lines where placeholders were damaged -> translate unmasked
    retry = []
    for _, _, items in parsed:
        for item in items:
            if isinstance(item, Segment) and item.codes:
                translated = cache.get(item.masked, "")
                if not translated or restore_code(translated, item.codes) is None:
                    retry.append(item.original)
    if retry:
        print(f"{len(set(retry))} lines lost code placeholders; retrying them unmasked.")
        translate_missing(retry, cache, backend, args.batch_size, "Pass 2")

    #Build the translated dataset
    stats = {"fallback": 0, "kept_english": 0}
    for idx, key, items in parsed:
        data[idx][key] = build_text(items, cache, stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)  # extra fail-safe
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Translated file created: {out_path}")
    print(f"Lines translated unmasked (fallback): {stats['fallback']}")
    print(f"Lines left in English:                {stats['kept_english']}")


if __name__ == "__main__":
    main()
