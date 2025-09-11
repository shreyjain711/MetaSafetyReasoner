import argparse

def main():
    parser = argparse.ArgumentParser(description="Process model name and prompt files.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--prompt_1', type=str, required=True, help='Filename for prompt 1')
    parser.add_argument('--prompt_2', type=str, required=True, help='Filename for prompt 2')
    args = parser.parse_args()

    print(f"Model Name: {args.model_name}")
    print(f"Prompt 1 File: {args.prompt_1}")
    print(f"Prompt 2 File: {args.prompt_2}")

if __name__ == "__main__":
    main()