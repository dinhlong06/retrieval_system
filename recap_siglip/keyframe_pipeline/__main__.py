from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_RECAP_MODEL,
    DEFAULT_SIGLIP_MODEL_ID,
    DEFAULT_SIGLIP_NUM_WORKERS,
    KeyframePipeline,
    KeyframePipelineError,
    extract_siglip_dataset,
    extract_siglip_from_dir,
    generate_recap_dataset,
    generate_recap_from_dir,
)


def _add_metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ocr-json", type=Path)
    parser.add_argument("--objects-json", type=Path)
    parser.add_argument("--resume", action="store_true")


def _add_ollama_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ollama-model", default=DEFAULT_RECAP_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--max-concurrency", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="keyframe-pipeline")
    parser.add_argument("--debug", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    siglip_video = commands.add_parser("siglip-video")
    siglip_video.add_argument("--keyframe-dir", type=Path, required=True)
    siglip_video.add_argument("--output-dir", type=Path, required=True)
    siglip_video.add_argument("--model-id", default=DEFAULT_SIGLIP_MODEL_ID)
    siglip_video.add_argument("--device")
    siglip_video.add_argument("--batch-size", type=int, default=32)
    siglip_video.add_argument("--num-workers", type=int, default=DEFAULT_SIGLIP_NUM_WORKERS)
    siglip_video.add_argument("--overwrite", action="store_true")

    siglip_dataset = commands.add_parser("siglip-dataset")
    siglip_dataset.add_argument("--dataset-root", type=Path, required=True)
    siglip_dataset.add_argument("--output-dir", type=Path, required=True)
    siglip_dataset.add_argument("--model-id", default=DEFAULT_SIGLIP_MODEL_ID)
    siglip_dataset.add_argument("--device")
    siglip_dataset.add_argument("--batch-size", type=int, default=32)
    siglip_dataset.add_argument("--num-workers", type=int, default=DEFAULT_SIGLIP_NUM_WORKERS)
    siglip_dataset.add_argument("--overwrite", action="store_true")

    recap_video = commands.add_parser("recap-video")
    recap_video.add_argument("--keyframe-dir", type=Path, required=True)
    recap_video.add_argument("--siglip-npy", type=Path)
    recap_video.add_argument("--siglip-ids", type=Path)
    recap_video.add_argument("--output", type=Path, required=True)
    recap_video.add_argument("--overwrite", action="store_true")
    _add_metadata_args(recap_video)
    _add_ollama_args(recap_video)

    recap_dataset = commands.add_parser("recap-dataset")
    recap_dataset.add_argument("--dataset-root", type=Path, required=True)
    recap_dataset.add_argument("--siglip-dir", type=Path)
    recap_dataset.add_argument("--output-dir", type=Path, required=True)
    recap_dataset.add_argument("--overwrite", action="store_true")
    _add_metadata_args(recap_dataset)
    _add_ollama_args(recap_dataset)

    run = commands.add_parser("run")
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL_ID)
    run.add_argument("--siglip-device")
    run.add_argument("--siglip-batch-size", type=int, default=32)
    run.add_argument("--overwrite", action="store_true")
    _add_metadata_args(run)
    _add_ollama_args(run)
    return parser


def _summary(kind: str, videos: int, keyframes: int, failed: int = 0) -> None:
    print(json.dumps({"kind": kind, "videos": videos, "keyframes": keyframes, "failed": failed}))


def _execute(args: argparse.Namespace) -> None:
    if args.command == "siglip-video":
        result = extract_siglip_from_dir(
            args.keyframe_dir,
            model_id=args.model_id,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            overwrite=args.overwrite,
        )
        _summary("siglip", 1, len(result.keyframe_ids))
    elif args.command == "siglip-dataset":
        result = extract_siglip_dataset(
            args.dataset_root,
            output_dir=args.output_dir,
            model_id=args.model_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            overwrite=args.overwrite,
        )
        _summary("siglip", len(result.results), sum(len(x.keyframe_ids) for x in result.results))
    elif args.command == "recap-video":
        result = generate_recap_from_dir(
            args.keyframe_dir,
            model=args.ollama_model,
            ollama_host=args.ollama_host,
            ocr_json_path=args.ocr_json,
            objects_json_path=args.objects_json,
            resume=args.resume,
            previous_embedding_path=args.siglip_npy,
            previous_ids_path=args.siglip_ids,
            output_path=args.output,
            max_concurrency=args.max_concurrency,
            overwrite=args.overwrite,
        )
        _summary("recap", 1, len(result.records), len(result.failures))
    elif args.command == "recap-dataset":
        result = generate_recap_dataset(
            args.dataset_root,
            output_dir=args.output_dir,
            model=args.ollama_model,
            ollama_host=args.ollama_host,
            ocr_json_path=args.ocr_json,
            objects_json_path=args.objects_json,
            resume=args.resume,
            previous_siglip_dir=args.siglip_dir,
            max_concurrency=args.max_concurrency,
            overwrite=args.overwrite,
        )
        _summary(
            "recap",
            len(result.results),
            sum(len(x.records) for x in result.results),
            sum(len(x.failures) for x in result.results),
        )
    else:
        pipeline = KeyframePipeline(
            siglip_model_id=args.siglip_model,
            siglip_device=args.siglip_device,
            recap_model=args.ollama_model,
            ollama_host=args.ollama_host,
        )
        siglip, recap = pipeline.run_dataset(
            args.dataset_root,
            output_root=args.output_root,
            ocr_json_path=args.ocr_json,
            objects_json_path=args.objects_json,
            siglip_batch_size=args.siglip_batch_size,
            recap_max_concurrency=args.max_concurrency,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        _summary(
            "pipeline",
            len(siglip.results),
            sum(len(x.records) for x in recap.results),
            sum(len(x.failures) for x in recap.results),
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _execute(args)
    except KeyframePipelineError as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

