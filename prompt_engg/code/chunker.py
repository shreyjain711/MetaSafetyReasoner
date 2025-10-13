import re
import json

# first split by \n\n, then by \n, then by .
def sent_splitter(text):
    paragraphs = text.split('\n\n')
    sentences = []
    for para in paragraphs:
        lines = para.split('\n')
        for line in lines:
            parts = re.split(r'(?<=[.!?]) +', line)
            sentences.extend(parts)
    return sentences

def para_splitter(text):
    paragraphs = text.split('\n\n')
    return paragraphs

def create_json_chunks(ls_texts, ls_inputs):
    ls_chunks = [sent_splitter(text) for text in ls_texts]
    ls_para_chunks = [para_splitter(text) for text in ls_texts]
    json_chunks = []
    for i, chunk in enumerate(ls_chunks):
        obj = {
            "input_id": f'{i}',
            "input": ls_inputs[i],
            "output": ls_texts[i],
            "chunks": [],
            "manual_sent_chunks": [],
            "manual_para_chunks": []
        }
        for j, sentence in enumerate(chunk):
            obj["chunks"].append({
                "chunk_id": f'ss_{i}_{j}',
                "text": sentence,
                "stance": "pro|neutral|against",
                "is_safe": 1,
                "is_complying": 1
            })
        for j, para in enumerate(ls_para_chunks[i]):
            obj["manual_para_chunks"].append({
                "chunk_id": f'mp_{i}_{j}',
                "text": para,
                "stance": "pro|neutral|against",
                "is_safe": 1,
                "is_complying": 1
            })
        obj["manual_sent_chunks"] = obj["chunks"]  # use the same for now, change later manually
        json_chunks.append(obj)

    return json_chunks

def get_input_by_dataset(data, dataset):
    if dataset == 'multi_turn_subset_224':
        return f"{data['objective']}\n{data['user_input']}"
    elif dataset == 'MSR_BeaverTails_4x56_subset':
        return data['prompt']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")



if __name__ == "__main__":
    inputs = []
    responses = []
    model = 'OpenThinker3-7B' # 'DeepSeek-R1-Distill-Qwen-7B'
    dataset = 'MSR_BeaverTails_4x56_subset' # 'multi_turn_subset_224'
    with open(f'/ocean/projects/cis250042p/sjain13/MetaSafetyReasoner/prompt_engg/outputs/results_{model}_{dataset}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            inputs.append(get_input_by_dataset(data, dataset))
            responses.append(data[f'response_{model}'][1])
    
    json_chunks = create_json_chunks(responses, inputs)
    
    with open(f'/ocean/projects/cis250042p/sjain13/MetaSafetyReasoner/prompt_engg/outputs/chunks_{model}_{dataset}.json', 'w') as f:
        for obj in json_chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')