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
    parser.add_argument('--prompt_path', type=str, required=True, help='Filename for scoring prompt')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=250, help='batch size/num of workers')
    parser.add_argument('--prompt_field', type=str, default='prompt', help='field name for the prompt in the dataset')
    args = parser.parse_args()

    output_file = f"outputs/results_{args.model_name.split('/')[-1]}_{args.data_file.split('.')[0].split('/')[-1]}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        # Process one batch at a time
        for batch in (data_generator(args.data_file, args.batch_size, args.prompt_path)):
            batch_data, batch_messages = zip(*batch)
            responses = batch_call_litellm(batch_messages, model=args.model_name, client=args.client, max_workers=args.batch_size)
            for data, response in zip(batch_data, responses):
                res = {**data, f'response_{args.model_name.split("/")[-1]}': response}
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

# python code/main.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8" --client=vllm --data_file="hf://Nishoak/MetaSafetyReasoner_Dataset" --batch_size=256 --score_prompt="/ocean/projects/cis250042p/sjain13/MetaSafetyReasoner/prompt_engg/prompts/saferbench_risk_cat.yml"