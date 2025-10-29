import os
import json
import openai
import litellm
from tqdm import tqdm
# from multiprocessing import Pool
import concurrent.futures
from validator import is_valid_response

litellm_client = None
API_KEY = None
CLIENT = None
VALIDATOR = None

def load_api_key(secrets_file, key_name='LITELLM_API_KEY'):
    with open(secrets_file, 'r') as f:
        secrets = json.load(f)
    return secrets.get(key_name)

def call_openai(messages, model):
    try:
        global API_KEY, CLIENT
        if API_KEY is None:
            API_KEY = load_api_key('secrets.json', key_name='OPENAI_API_KEY')
        if CLIENT is None:
            CLIENT = openai.OpenAI(api_key=API_KEY)

        response = CLIENT.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

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

def call_vllm(messages, model, PORT):
    global API_KEY, CLIENT, VALIDATOR
    if API_KEY is None:
        API_KEY = load_api_key('secrets.json', key_name='VLLM_API_KEY')
    if CLIENT is None:
        # default URL for the vLLM server
        base_url = os.getenv("VLLM_BASE_URL", f"http://localhost:{PORT}/v1")
        CLIENT = openai.OpenAI(api_key=API_KEY, base_url=base_url)
    tries, response = 0, None
    while tries < 5 and response is None:
        try:
            tries += 1
            response = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512
            )
            response = response.choices[0].message.content
            if is_valid_response(response.split('</think>')[-1], VALIDATOR) is False:
                response = None
            else: response = response.split('</think>')[-1]
        except Exception as e:
            
            response = None
    return response
    
def _call_client_wrapper(args):
    i, messages, model, client, port = args
    if client == "litellm":
        return i, call_litellm(messages, model)
    elif client == "vllm":
        return i, call_vllm(messages, model, port)
    elif client == "openai":
        return i, call_openai(messages, model)
    else:
        raise ValueError(f"Invalid client: {client}")


#def batch_call_model(batch_messages, model="openai/gpt-4o", client="litellm", secrets_file='secrets.json', max_workers=5, port=11632, validator=None):
#    """
#    batch_messages: list of list of messages, e.g. [[{"role": "user", "content": "Hi"}], ...]
#    Returns: list of responses
#    """
#    global API_KEY, VALIDATOR
#    if validator is not None:
#        VALIDATOR = validator
#    
#    args_list = [(i, message, model, client, port) for i, message in enumerate(batch_messages)]
#    results = []
#    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#        futures = [executor.submit(_call_client_wrapper, args) for args in args_list]
#        for future in tqdm(concurrent.futures.as_completed(futures), total=len(batch_messages), desc="Processing batch"):
#            results.append(future.result())
#    return [r[1] for r in sorted(results, key=lambda x: x[0])]

def batch_call_model(batch_messages, model="openai/gpt-4o", client="litellm", secrets_file='secrets.json', max_workers=5, port=11632, validator=None):
    """
    batch_messages: list of list of messages, e.g. [[{"role": "user", "content": "Hi"}], ...]
    Returns: list of responses
    """
    global API_KEY, VALIDATOR
    if validator is not None:
        VALIDATOR = validator

    args_list = [(i, message, model, client, port) for i, message in enumerate(batch_messages)]
    results = {}  # Using dict for O(1) insertion
    futures_map = {}  # Map future objects to their indices
    retry_count = {}  # Track retries per index
    max_retries = 3

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(batch_messages), desc="Processing batch") as pbar:
            active_futures = set()
            next_idx = 0
            optimal_batch = 800  # Match your API's optimal batch size

            while next_idx < len(args_list) or active_futures:
                # Submit tasks until we reach optimal_batch or max_workers
                while len(active_futures) < min(optimal_batch, max_workers) and next_idx < len(args_list):
                    future = executor.submit(_call_client_wrapper, args_list[next_idx])
                    active_futures.add(future)
                    futures_map[future] = next_idx
                    next_idx += 1

                # Check for completed futures
                done, active_futures = concurrent.futures.wait(
                    active_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                # Process completed futures
                for future in done:
                    idx = futures_map[future]
                    try:
                        result_idx, result = future.result()
                        results[result_idx] = result
                        pbar.update(1)
                    except Exception as e:
                        # Retry failed tasks up to max_retries
                        retry_count[idx] = retry_count.get(idx, 0) + 1
                        if retry_count[idx] <= max_retries:
                            new_future = executor.submit(_call_client_wrapper, args_list[idx])
                            active_futures.add(new_future)
                            futures_map[new_future] = idx
                        else:
                            results[idx] = f"Error: {e} (max retries exceeded)"
                            pbar.update(1)

    # Fill missing results with error message
    sorted_results = [results[i] if i in results else None for i in range(len(batch_messages))]
    return sorted_results