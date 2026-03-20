import os
import gc
import re
import torch
import pickle
from glob import glob
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# os.environ['HF_HOME'] = os.getenv('PROJECT')
# login(token=os.getenv('HF_TOKEN'))


# MODEL_PATH = "/data/user_data/jamesdin/ckpts/MetaSafetyReasoner-RL/qwen3_0.6b_msr_step500_hf"
MODEL_PATH = "Qwen3-0.6B"

#########################
# Inference Functions  #
#########################


def get_ft_model(model_name, checkpoint_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              trust_remote_code=True,
                                              padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        print(f"Model loaded from {checkpoint_path}")

    return model, tokenizer

def get_msr_model(checkpoint_path):
    REPO_ID = "jamesding0302/MetaSafetyReasoner-RL"
    SUBFOLDER = f"qwen3_0.6b_msr_step{checkpoint_path}_hf"
    
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID,
        subfolder=SUBFOLDER,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        subfolder=SUBFOLDER,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map='auto',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def unpack_batch(batch_dict):
    """
    Convert a batch like:
      {"question": [...], "answer": [...]}
    into:
      [{"question": ..., "answer": ...}, ...]
    """
    keys = list(batch_dict.keys())
    n = len(batch_dict[keys[0]])
    return [{k: batch_dict[k][i] for k in keys} for i in range(n)]


def create_prompt_message(example, prompt_field):
    chat = []
    if 'system' in example: chat.append({'role': 'system', 'content': example['system']})
    chat.append({"role": "user", "content": example[prompt_field]})
    return chat


def batched_inference(model_name, model, tokenizer, dataset, prompt_field='prompt', batch_size=8, max_new_tokens=512):
    results = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = unpack_batch(dataset[i:i+batch_size])
        
        # Prepare chat-formatted inputs
        messages_batch = [create_prompt_message(ex, prompt_field) for ex in batch]

        # Tokenize with the chat template
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(model.device)

        sampling_params = {"do_sample": True}
        if model_name == 'meta-llama/Llama-3.1-8B-Instruct':
            sampling_params.update({
                "temperature": 0.6,
                "top_p": 0.9,
            })
        else: #if model_name == 'Qwen/Qwen3-4B-Instruct-2507':
            sampling_params.update({
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
            })

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **sampling_params
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Extract generated answers based on '####'
        for ex, user_msg, full_output in zip(batch, messages_batch, decoded_outputs):
            if "####" in full_output:
                generated = full_output.split("####")[-1].strip('# \n')
            else:
                # Fallback if the marker is missing
                generated = full_output.strip()
            user_msg.append({'role': 'assistant', 'content': generated})
            
            results.append({
                "question": ex[prompt_field],
                "ground_truth": ex.get('response', None),
                "chat": user_msg,
                "source_dataset": ex['source_dataset'],
            })

    return results

def run_inf(base_model, checkpoint_path, dataset, prompt_field='prompt', batch_size=8, max_new_tokens=512):
    model, tokenizer = get_msr_model(checkpoint_path) if checkpoint_path is not None else get_ft_model(base_model, checkpoint_path)
    print("DBG", base_model, checkpoint_path, 'is loaded')
    results = batched_inference(base_model, model, tokenizer, dataset, 
                                    prompt_field=prompt_field, 
                                    batch_size=batch_size, 
                                    max_new_tokens=max_new_tokens)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


#######################
# Main Evaluation  #
#######################

def extract_ans(text):
    return text.split('#')[-1].strip()

def eval_str_match(results, preproc_fn, verbose=False):
    """
    Evaluates model outputs for exact-match correctness.
    
    Returns:
        accuracy (float): percentage of correct answers
        correct_count (int)
        total_count (int)
    """
    correct = 0
    total = len(results)

    for i, r in enumerate(results):
        gt = preproc_fn(str(r["ground_truth"]))
        pred = preproc_fn(str(r['chat'][-1]['content']))

        if pred == gt:
            correct += 1

        if verbose and i%100==0:
            print(f"Q: {r['question']}")
            print(f"GT: {str(r['ground_truth'])} | Pred: {r['chat'][-1]['content']}")
            print("✅ Correct\n" if pred==gt else "❌ Incorrect\n")

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def eval_safety(guard_model, tokenizer, data, batch_size=128):
    all_decoded_outputs = []
    
    # Process the data in batches
    for i in tqdm(range(0, len(data), batch_size)):
        
        # 1. Get the current batch
        batch_data = data[i : i + batch_size]
        
        # Extract chats from the batch data
        batch_chats = [x['chat'] for x in batch_data]

        # 2. Tokenize the batch
        # Note: Using guard_model.device instead of model.device
        inputs = tokenizer.apply_chat_template(
            batch_chats,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(guard_model.device) 

        # 3. Generate outputs for the batch
        with torch.no_grad():
            outputs = guard_model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_decoded_outputs.extend(decoded_outputs)

        del inputs
        del outputs
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Calculate the final score across all collected outputs
    if not all_decoded_outputs:
        print("Warning: No data was processed.")
        return 0.0

    safe_count = sum([1 for o in all_decoded_outputs if o.split('\n\n')[-1] == 'safe'])
    total_count = len(all_decoded_outputs)

    return all_decoded_outputs, safe_count / total_count



if __name__ == "__main__":
    ### Inference
    LLAMA_PATTERN = '/ocean/projects/cis250042p/sjain13/IntrospectLoss/outputs/*_l3_8_alpha_[0|1]/checkpoint-*'
    # QWEN3_4_PATTERN = '/ocean/projects/cis250042p/sjain13/IntrospectLoss/training_POC/dbg_gsm_q3_4_alpha_*/checkpoint-234'

    models = {
        '/data/user_data/jamesdin/models/Qwen3-0.6B': [None],
        'msr_model_qwen': ['500', '1000', '1500']
    }

    for model, cps in models.items():
        print(f"Paths for {model}:")
        for cp in cps:
            print(f"    {cp if cp is not None else 'Base'}")

    eval_set = load_dataset('json', data_files='llm_eval_set.jsonl')

    util_sources = ["ARC_C", "GSM8K_Test"]
    safety_sources = ["AdvBench", "StrongREJECT", "ClearHarm"]

    util_set = eval_set['train'].filter(lambda example: example['source_dataset'] in util_sources)
    safety_set = eval_set['train'].filter(lambda example: example['source_dataset'] in safety_sources)

    pickle_path = 'temp_pkls'
    os.makedirs(pickle_path, exist_ok=True)

    # save the generated responses and results for util set
    util_infs = {}
    if 'util_infs.pkl' in os.listdir(pickle_path):
        with open(os.path.join(pickle_path, 'util_infs.pkl'), 'rb') as f:
            util_infs = pickle.load(f)
    
    for base_model, checkpoint_paths in models.items():
        if base_model not in util_infs: util_infs[base_model] = {}
        for cp in checkpoint_paths:
            exp_name = cp if cp is not None else 'Base'
            if base_model in util_infs and exp_name in util_infs[base_model].keys():
                print(f"Skipping {base_model} {exp_name}, already done.")
                continue
            print("Running util_set inference for", base_model, cp if cp is not None else 'Base')
            util_infs[base_model][exp_name] = run_inf(base_model, cp, util_set, batch_size=512, max_new_tokens=384)
            with open(os.path.join(pickle_path, 'util_infs.pkl'), 'wb') as f:
                pickle.dump(util_infs, f)
    
    safety_infs = {}
    if 'safety_infs.pkl' in os.listdir(pickle_path):
        with open(os.path.join(pickle_path, 'safety_infs.pkl'), 'rb') as f:
            safety_infs = pickle.load(f)
    
    for base_model, checkpoint_paths in models.items():
        if base_model not in safety_infs: safety_infs[base_model] = {}
        for cp in checkpoint_paths:
            exp_name = cp if cp is not None else 'Base'
            if base_model in safety_infs and exp_name in safety_infs[base_model]:
                print(f"Skipping {base_model} {exp_name}, already done.")
                continue
            print("Running safety_set inference for", base_model, cp if cp is not None else 'Base')
            safety_infs[base_model][cp if cp is not None else 'Base'] = run_inf(base_model, cp, safety_set, 
                                                                            batch_size=128, 
                                                                            max_new_tokens=768)
            with open(os.path.join(pickle_path, 'safety_infs.pkl'), 'wb') as f:
                pickle.dump(safety_infs, f)
    
    
    ### Util Eval
    results = {}
    for model, m_res in util_infs.items():
        results[model] = {}
        for exp_name, e_res in m_res.items():
            if '/' in exp_name: exp_name = exp_name.split('/')[-2]
            results[model][exp_name] = {}
            for src in util_sources:
                results[model][exp_name][src] = eval_str_match([e for e in e_res if e['source_dataset']==src], extract_ans, False)
            # print("DBG", src, [f"{src}: {results[model][exp_name][src]:.4f}" for src in util_sources])
            # print("DBG", src, [f"{src}: {results[model][exp_name][src]['accuracy']}" for src in util_sources])
            scores_str = "   ".join([f"{src}: {results[model][exp_name][src]['accuracy']:.4f}" for src in util_sources])
            print(f'{model:40} {exp_name:30} {scores_str}')

    ### Safety Eval 
    g_model, g_tokenizer = get_ft_model("/data/user_data/jamesdin/models/Llama-Guard-3-8B", None)
    
    if 'safety_results.pkl' in os.listdir(pickle_path):
        with open(os.path.join(pickle_path, 'safety_results.pkl'), 'rb') as f:
            safety_results = pickle.load(f)
    else:
        safety_results = {}
        for model, m_res in safety_infs.items():
            safety_results[model] = {}
            for exp_name, e_res in m_res.items():
                if '/' in exp_name: exp_name = exp_name.split('/')[-2]
                safety_results[model][exp_name] = {}
                for src in safety_sources:
                    safety_results[model][exp_name][src] = eval_safety(g_model, g_tokenizer,
                                                                       [e for e in e_res if e['source_dataset']==src], 64)
        with open(os.path.join(pickle_path, 'safety_results.pkl'), 'wb') as f:
            pickle.dump(safety_results, f)

    for model, m_res in safety_results.items():
        for exp_name, e_res in m_res.items():
            scores_str = "   ".join([f"{src}: {e_res[src][1]:.4f}" for src in safety_sources])
            print(f'{model:40} {exp_name:30} {scores_str}')