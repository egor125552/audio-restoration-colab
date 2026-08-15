from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--language", default="Russian")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from qwen_asr import Qwen3ASRModel
    from tqdm.auto import tqdm

    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=4,
        max_new_tokens=512,
    )

    out = []
    for item in tqdm(manifest, desc="Расшифровка", unit="фрагмент", dynamic_ncols=False, ncols=100, leave=False):
        result = model.transcribe(audio=item["audio"], language=args.language)[0]
        text = (result.text or "").strip()
        out.append({**item, "text": text, "language": getattr(result, "language", args.language)})

    Path(args.output).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
