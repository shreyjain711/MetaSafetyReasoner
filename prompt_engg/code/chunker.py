import re
import json


def para_splitter(text):
    paragraphs = text.split('\n\n')
    paragraphs = [para + '\n\n'  if i < len(paragraphs) - 1 else para for i, para in enumerate(paragraphs)]
    return paragraphs

# first split by \n\n, then by \n, then by .
def sent_splitter(text):
    paragraphs = para_splitter(text)
    sentences = []
    for para in paragraphs:
        lines = para.split('\n')
        for line in lines:
            parts = re.split(r'(?<=[.!?]) +', line)
            sentences.extend(parts)
    return sentences

import re

# split text by paragraph (\n\n), then by line (\n), and finally by sentence; preserving delimiters
def sent_splitter_preserve_delimiters(text):
    para_parts = re.split(r'(\n{2,})', text)
    paragraphs_with_delimiters = []
    
    # Merge text with its delimiter: [text1, \n\n, text2, \n\n, text3] -> ['text1\n\n', 'text2\n\n', 'text3']
    for i in range(0, len(para_parts), 2):
        if i + 1 < len(para_parts):
            paragraphs_with_delimiters.append(para_parts[i] + para_parts[i+1])
        else:
            if para_parts[i].strip():
                 paragraphs_with_delimiters.append(para_parts[i])

    # 2. Line Splitting and Sentence Splitting
    final_sentences = []
    for para in paragraphs_with_delimiters:
        # Check if the paragraph is just a delimiter (which can happen with leading delimiters)
        if not para.strip():
            continue
            
        # a. Line Splitting
        # Pattern: Matches and captures a single newline (\n).
        line_parts = re.split(r'(\n)', para)
        lines_with_delimiters = []
        
        # Merge text with its delimiter
        for i in range(0, len(line_parts), 2):
            text_chunk = line_parts[i]
            if i + 1 < len(line_parts):
                # Merge the text at i with the delimiter at i+1
                text_chunk += line_parts[i+1]
            
            # b. Sentence Splitting
            # Pattern: Matches a sentence-ending punctuation (., !, ?) followed by one or more spaces, and captures BOTH.
            # This ensures the entire delimiter is kept together.
            # The lookbehind `(?<=[.!?])` from your original is replaced by a capturing group `([.!?]+)\s+`.
            sentence_parts = re.split(r'([.!?]+)\s+', text_chunk)
            
            # Merge sentence text with its delimiter
            for j in range(0, len(sentence_parts), 2):
                if sentence_parts[j].strip(): # Ensure the text part isn't empty
                    sentence_with_delimiter = sentence_parts[j]
                    if j + 1 < len(sentence_parts):
                        # Merge text at j with delimiter (punctuation + space) at j+1
                        sentence_with_delimiter += sentence_parts[j+1] + ' '
                    
                    final_sentences.append(sentence_with_delimiter)

    return final_sentences

def create_json_chunks(ls_texts, ls_inputs):
    ls_para_chunks = [para_splitter(text) for text in ls_texts]
    ls_chunks = [sent_splitter_preserve_delimiters(text) for text in ls_texts]
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
    model = 'Qwen3-30B-A3B-Instruct-2507-FP8' # 'OpenThinker3-7B' # 'DeepSeek-R1-Distill-Qwen-7B'
    dataset = 'multi_turn_subset_224' # 'MSR_BeaverTails_4x56_subset'
    with open(f'/ocean/projects/cis250042p/sjain13/MetaSafetyReasoner/prompt_engg/outputs/results_{model}_{dataset}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            inputs.append(get_input_by_dataset(data, dataset))
            responses.append(data[f'response_{model}'][1])
    
    json_chunks = create_json_chunks(responses, inputs)
    
    with open(f'/ocean/projects/cis250042p/sjain13/MetaSafetyReasoner/prompt_engg/outputs/chunks_{model}_{dataset}.json', 'w') as f:
        for obj in json_chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')