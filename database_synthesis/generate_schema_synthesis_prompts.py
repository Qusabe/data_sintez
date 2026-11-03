import json
import os
import random
import numpy as np

# ---------- настройки пробного прогона ----------
NUM_SOURCE_TABLES = 80    # сколько веб-таблиц взять из 19 935 (≈ столько баз получится)
TABLES_MEAN = 10          # среднее число таблиц в одной базе (в оригинале 10)
TABLES_STD = 4          # разброс (в оригинале 4)
TABLES_MIN = 1            # минимум таблиц в базе (в оригинале 1)
TABLES_MAX = 20            # максимум таблиц в базе (в оригинале 20)
# ------------------------------------------------

def generate_a_normal_integer(mean=TABLES_MEAN, std_dev=TABLES_STD,
                              lower_bound=TABLES_MIN, upper_bound=TABLES_MAX):
    sample = np.random.normal(mean, std_dev)
    sample = np.clip(sample, lower_bound, upper_bound)
    return int(round(sample))

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)   # random.seed не влияет на numpy, без этого числа таблиц меняются при каждом запуске

    tables = json.load(open("web_tables.json", "r", encoding="utf-8"))
    prompt_template = open("database_synthesis/prompt_templates/schema_prompt.txt", "r", encoding="utf-8").read()

    # ограничение №1: берём только часть исходных веб-таблиц
    tables = random.sample(tables, min(NUM_SOURCE_TABLES, len(tables)))

    prompts = []
    for table in tables:
        # ограничение №2: число таблиц в создаваемой базе
        random_table_num = generate_a_normal_integer()
        prompt = prompt_template.format(table_num=random_table_num, table=table)
        prompts.append(prompt.strip())

    random.shuffle(prompts)

    os.makedirs("database_synthesis/prompts", exist_ok=True)
    with open("database_synthesis/prompts/prompts_schema_synthesis.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(prompts, ensure_ascii=False, indent=2))

    print(f"Готово: {len(prompts)} промптов, таблиц в базе от {TABLES_MIN} до {TABLES_MAX}")