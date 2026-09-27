
from openai import OpenAI
from argparse import ArgumentParser

from torch import chunk


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3.8-27B",
        help="The model to use for the response.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run the script in interactive mode.",
    )
    return parser.parse_args()

def run_interactive_mode(client, model):
    print("Entering interactive mode. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
        streamer = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        print("Assistant: ", end="", flush=True)
        raw = []
        for chunk in streamer:
            message_response = chunk.choices[0].delta.content
            if message_response:
                print(message_response, end="", flush=True)
            raw.append(chunk)
        print()  # Newline after the response
    
        
def main():

    args = parse_args()
    
    client = OpenAI(
        api_key="placeholder",
        base_url="http://localhost:8000/v1",
    )

    if args.interactive:
        run_interactive_mode(client, args.model)
        return
    
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
