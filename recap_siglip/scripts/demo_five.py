"""Run the real SigLIP + Ollama pipeline on a small keyframe subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Keep the demo directly runnable without requiring an editable install first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keyframe_pipeline import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_RECAP_MODEL,
    DEFAULT_SIGLIP_MODEL_ID,
    discover_video,
    extract_siglip,
    generate_recap,
    load_caption_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--ocr-json", type=Path)
    parser.add_argument("--objects-json", type=Path)
    parser.add_argument("--stage", choices=("all", "siglip", "recap"), default="all")
    parser.add_argument("--device", default=None, help="For example: cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL_ID)
    parser.add_argument("--recap-model", default=DEFAULT_RECAP_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    video = discover_video(args.keyframe_dir)
    keyframes = video.keyframes[: args.limit]
    if not keyframes:
        raise SystemExit("No keyframes found")

    metadata = load_caption_metadata(
        ocr_json_path=args.ocr_json,
        objects_json_path=args.objects_json,
    )
    siglip_dir = args.output_dir / "siglip"

    siglip_result = None
    if args.stage in ("all", "siglip"):
        siglip_result = extract_siglip(
            keyframes,
            model_id=args.siglip_model,
            output_dir=siglip_dir,
            batch_size=args.batch_size,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "stage": "siglip",
                    "video_id": siglip_result.video_id,
                    "shape": list(siglip_result.embeddings.shape),
                    "dtype": str(siglip_result.embeddings.dtype),
                    "norm_min": float(np.linalg.norm(siglip_result.embeddings, axis=1).min()),
                    "norm_max": float(np.linalg.norm(siglip_result.embeddings, axis=1).max()),
                    "embedding_path": str(siglip_result.embedding_path),
                    "ids_path": str(siglip_result.ids_path),
                },
                ensure_ascii=False,
            )
        )

    if args.stage in ("all", "recap"):
        recap_path = args.output_dir / "recap" / f"{video.video_id}.jsonl"
        recap_result = generate_recap(
            keyframes,
            model=args.recap_model,
            ollama_host=args.ollama_host,
            metadata=metadata,
            previous_output=siglip_result,
            output_path=recap_path,
            max_concurrency=1,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "stage": "recap",
                    "video_id": recap_result.video_id,
                    "count": len(recap_result.records),
                    "output_path": str(recap_result.output_path),
                    "captions": [record.caption for record in recap_result.records],
                    "corrected_ocr": [
                        list(record.corrected_ocr) for record in recap_result.records
                    ],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
