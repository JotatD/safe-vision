"""Score existing OpenFlamingo responses with Gemini."""
import argparse
import json
import os
from pathlib import Path

import time

from eval_openflamingo_vljailbreak import (
    GeminiSafetyJudge,
    VLJailbreakSample,
)


def load_samples(jsonl_path: Path):
    with jsonl_path.open() as fh:
        for line in fh:
            if line.strip():
                data = json.loads(line)
                yield VLJailbreakSample(
                    idx=data.get("idx", data.get("index")),
                    prompt=data.get("prompt", data.get("text_prompt", "")),
                    image_path=Path(data.get("image_path", data.get("image", ""))),
                    category=data.get("category", "Unknown"),
                    subcategory=data.get("subcategory", "Unknown"),
                    goal=data.get("goal"),
                    metadata={k: v for k, v in data.items() if k not in {"idx", "index", "prompt", "text_prompt", "image_path", "image", "category", "subcategory", "goal", "response", "harmful"}},
                    response=data.get("response"),
                    harmful=data.get("harmful"),
                )


def main():
    parser = argparse.ArgumentParser(description="Score existing results with Gemini")
    parser.add_argument("--input", required=True, help="Path to openflamingo_vljailbreak_results.jsonl")
    parser.add_argument("--output", required=True, help="Where to write scored JSONL")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash-lite")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY environment variable is required")

    judge = GeminiSafetyJudge(model_name=args.gemini_model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as out_fh:
        for sample in load_samples(input_path):
            if not sample.response:
                sample.harmful = "error"
            else:
                # Simple rate-limit handling with retry/backoff
                for attempt in range(5):
                    try:
                        sample.harmful = judge.score(sample.response)
                        time.sleep(0.5)
                        break
                    except Exception as exc:
                        if "429" in str(exc) and attempt < 4:
                            sleep = 2 ** attempt
                            print(f"[Gemini] 429 rate limit, retrying in {sleep}s...")
                            time.sleep(sleep)
                            continue
                        print(f"[Gemini] Error while scoring response: {exc}")
                        sample.harmful = "error"
                        break
            out_fh.write(json.dumps(sample.to_export_dict()) + "\n")


if __name__ == "__main__":
    main()
