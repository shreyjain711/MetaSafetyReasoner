import os
import json
import argparse
from tqdm import tqdm
from model_client import batch_call_litellm
from data_generator import data_generator

def main():
    parser = argparse.ArgumentParser(description="Process model name and prompt files.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--client', type=str, required=True, help='Name of the client. e.g. litellm, vllm, openai.')
    parser.add_argument('--score_prompt', type=str, required=True, help='Filename for scoring prompt')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=250, help='batch size/num of workers')
    args = parser.parse_args()

    results = []
    for batch in tqdm(data_generator(args.data_file, args.batch_size)):
        batch_messages = [[{"role": "user", "content": d['prompt']}] for d in batch]
        responses = batch_call_litellm(batch_messages, model=args.model_name, client=args.client, max_workers=args.batch_size)
        for data, response in zip(batch, responses):
            results.append({**data, f'response_{args.model_name.split("/")[-1]}': response})

    output_file = f"outputs/results_{args.model_name.split('/')[-1]}_{args.data_file.split('.')[0].split('/')[-1]}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()