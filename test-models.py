#!/usr/bin/env python3

import time
from openai import OpenAI

BASE_URL = "https://kaggle-inference-proxy.onrender.com/v1"
API_KEY = "sk-change-me-client-key"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

PROMPT = """Write exactly one short sentence about artificial intelligence."""

def get_models():
    models = client.models.list()
    return [m.id for m in models.data]


def test_model(model_name):
    print("=" * 80)
    print(f"Testing: {model_name}")

    first_token_time = None
    total_text = ""

    start = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                }
            ],
            temperature=0.7,
            max_tokens=64,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""

            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                print(delta, end="", flush=True)
                total_text += delta

        end = time.perf_counter()

        print("\n")

        if first_token_time:
            ttft = first_token_time - start
            total = end - start

            print(f"✓ Streaming: YES")
            print(f"TTFT:        {ttft:.3f}s")
            print(f"Total Time:  {total:.3f}s")
            print(f"Chars:       {len(total_text)}")

            if total > ttft:
                rate = len(total_text) / (total - ttft)
                print(f"Char/sec:    {rate:.2f}")

        else:
            print("⚠ No streamed tokens received.")

    except Exception as e:
        print(f"✗ Failed")
        print(e)

    print()


def main():
    print("Fetching models...\n")

    models = get_models()

    print(f"Found {len(models)} models\n")

    for model in models:
        test_model(model)


if __name__ == "__main__":
    main()