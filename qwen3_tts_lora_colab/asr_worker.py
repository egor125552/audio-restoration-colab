from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--language", default="Russian")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    batch_size = max(1, args.batch_size)

    # Keep the notebook readable: these are dependency/generation warnings, not
    # ASR failures. Real exceptions still propagate normally.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    import torch
    from qwen_asr import Qwen3ASRModel
    from tqdm.auto import tqdm

    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=batch_size,
        max_new_tokens=512,
    )

    out = []
    with tqdm(total=len(manifest), desc="Расшифровка", unit="фрагмент", dynamic_ncols=False, ncols=100, leave=False) as progress:
        for start in range(0, len(manifest), batch_size):
            batch = manifest[start:start + batch_size]
            audio_batch = [item["audio"] for item in batch]
            results = model.transcribe(audio=audio_batch, language=args.language)
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Qwen3-ASR вернул {len(results)} результатов для пакета из {len(batch)} фрагментов."
                )
            for item, result in zip(batch, results):
                text = (result.text or "").strip()
                out.append({**item, "text": text, "language": getattr(result, "language", args.language)})
            progress.update(len(batch))

    Path(args.output).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
