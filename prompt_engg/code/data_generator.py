import yaml
import json
from datasets import load_dataset

def create_message(data, user_prompt, system_prompt):
    if system_prompt is None:
        return [{"role": "user", "content": user_prompt}]
    return [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]

def get_promt_from_file(prompt_file_path):
    if prompt_file_path.endswith('.txt'):
        with open(prompt_file_path, 'r') as f:
            return (f.read().strip(), None)
    elif prompt_file_path.endswith('.yml'):
        with open(prompt_file_path, 'r') as f:
            prompt_data = yaml.safe_load(f)
            return (prompt_data.get('prompts').get('system').strip(), prompt_data.get('prompts').get('user').strip())


def dataset_reader(file_path):
    if file_path.startswith('hf://'):
        dataset = load_dataset(file_path[5:])
        for item in dataset['train']:
            yield item
    else:
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)


def data_generator(file_path, batch_size, prompt_file_path=None):
    user_prompt, system_prompt = get_promt_from_file(prompt_file_path)
    batch = []
    for data in dataset_reader(file_path):
        batch.append((data, create_message(data, user_prompt, system_prompt)))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch