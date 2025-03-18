import json
from collections import defaultdict


category_dict = defaultdict(list)


def traverse_file(data, path=None, result=None):
    if path is None:
        path = []
    if result is None:
        result = []

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for example in value:
                    if isinstance(example, dict) and "prompt_text" in example:
                        prompt_taxonomy_label = " - ".join(path + [key])
                        prompt_text = example["prompt_text"]
                        result.append({"prompt_taxonomy_label": prompt_taxonomy_label,
                                       "prompt_id": example["prompt_id"],
                                       "prompt_text": prompt_text})
                        category_dict[prompt_taxonomy_label].append(example["prompt_id"])
            traverse_file(value, path + [key], result)
    elif isinstance(data, list):
        for item in data:
            traverse_file(item, path, result)

    return result


if __name__ == '__main__':
    with open("ipv_txt_prompt_suite.json", 'r') as f:
        data = json.load(f)
    prompt_list = traverse_file(data)
    print(f"{len(prompt_list)} prompts in total.")
    print("Category stats:")
    for key, value in sorted(category_dict.items()):
        print(f"{key}: {len(value)}")

