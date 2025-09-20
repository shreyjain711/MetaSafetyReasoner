import os
import json
import litellm
from tqdm import tqdm
from multiprocessing import Pool

litellm_client = None
API_KEY = None

def load_api_key(secrets_file):
    with open(secrets_file, 'r') as f:
        secrets = json.load(f)
    return secrets.get('LITELLM_API_KEY')

def call_litellm(messages, model):
    try:
        global API_KEY
        if API_KEY is None:
            API_KEY = load_api_key('secrets.json')
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=API_KEY,
            base_url="https://cmu.litellm.ai"
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def _call_litellm_wrapper(args):
    messages, model = args
    return call_litellm(messages, model)

# Example usage:
# messages = [{"role": "user", "content": "Hello, who are you?"}]
# print(call_litellm(messages))

def batch_call_litellm(batch_messages, model="openai/gpt-4o", secrets_file='secrets.json', max_workers=5):
    """
    batch_messages: list of list of messages, e.g. [[{"role": "user", "content": "Hi"}], ...]
    Returns: list of responses
    """
    global API_KEY
    API_KEY = load_api_key(secrets_file)
    args_list = [(message, model) for message in batch_messages]
    results = []
    with Pool(processes=max_workers) as pool:
        for result in tqdm(pool.imap_unordered(_call_litellm_wrapper, args_list), total=len(batch_messages), desc="Processing batch"):
            results.append(result)
    return results