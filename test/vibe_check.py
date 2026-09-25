
from openai import OpenAI
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3.8-27B",
        help="The model to use for the response.",
    )
    return parser.parse_args()

def main():

    args = parse_args()
    
    client = OpenAI(
        api_key="placeholder",
        base_url="http://localhost:8000/v1",
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 967 + 32?"},
    ]
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
    )

    output = response.choices[0].message.content
    print(f"Model: {args.model}")
    print(f"Response: {output}")
    

if __name__ == "__main__":
    main()
