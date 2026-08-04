"""
cli.py — Command Line Interface cho Keyframe Extractor Benchmark.

Sử dụng:
  python cli.py --pipeline pipeline_c --video_dir dataset/raw_video --shots_dir dataset/shots

  hoặc dùng file config yaml:
  python cli.py --config configs/pipeline_c.yaml --video_dir dataset/raw_video

  benchmark nhiều pipeline cùng lúc:
  python cli.py --pipeline all --video_dir dataset/raw_video

Toàn bộ tham số cũng có thể override qua CLI flag sau khi chỉ định --config.
"""

import argparse
import sys
from pathlib import Path
import yaml


def load_config(config_path: str) -> dict:
    """
    Đọc file YAML config và trả về dict.

    Args:
        config_path : Đường dẫn đến file .yaml.

    Returns:
        Dict chứa cấu hình keyframe pipeline.
    """
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline_a(cfg: dict, args: argparse.Namespace):
    """Khởi tạo PipelineA từ config + CLI args."""
    from src.extractors.pipeline_a import PipelineA
    beit3_cfg = cfg.get("beit3", {})
    semantic_cfg = cfg.get("semantic", {})
    return PipelineA(
        checkpoint_path=args.checkpoint_path or beit3_cfg.get("checkpoint_path", ""),
        spm_path=args.spm_path or beit3_cfg.get("spm_path", ""),
        similarity_threshold=args.threshold or semantic_cfg.get("similarity_threshold", 0.90),
        device=args.device or beit3_cfg.get("device"),
        batch_size=args.batch_size or beit3_cfg.get("batch_size", 32),
    )


def build_pipeline_b(cfg: dict, args: argparse.Namespace):
    """Khởi tạo PipelineB từ config + CLI args."""
    from src.extractors.pipeline_b import PipelineB
    mobile_cfg = cfg.get("mobilenet", {})
    semantic_cfg = cfg.get("semantic", {})
    return PipelineB(
        similarity_threshold=args.threshold or semantic_cfg.get("similarity_threshold", 0.90),
        device=args.device or mobile_cfg.get("device"),
        batch_size=args.batch_size or mobile_cfg.get("batch_size", 64),
    )


def build_pipeline_c(cfg: dict, args: argparse.Namespace):
    """Khởi tạo PipelineC từ config + CLI args."""
    from src.extractors.pipeline_c import PipelineC
    beit3_cfg = cfg.get("beit3", {})
    dake_cfg = cfg.get("dake", {})
    semantic_cfg = cfg.get("semantic", {})
    return PipelineC(
        checkpoint_path=args.checkpoint_path or beit3_cfg.get("checkpoint_path", ""),
        spm_path=args.spm_path or beit3_cfg.get("spm_path", ""),
        candidate_ratio=args.candidate_ratio or dake_cfg.get("candidate_ratio", 0.02),
        window_size=dake_cfg.get("window_size", 3),
        similarity_threshold=args.threshold or semantic_cfg.get("similarity_threshold", 0.90),
        device=args.device or beit3_cfg.get("device"),
        batch_size=args.batch_size or beit3_cfg.get("batch_size", 32),
    )


def build_pipeline_d(cfg: dict, args: argparse.Namespace):
    """Khởi tạo PipelineD từ config + CLI args."""
    from src.extractors.pipeline_d import PipelineD
    mobile_cfg = cfg.get("mobilenet", {})
    dake_cfg = cfg.get("dake", {})
    semantic_cfg = cfg.get("semantic", {})
    return PipelineD(
        candidate_ratio=args.candidate_ratio or dake_cfg.get("candidate_ratio", 0.02),
        window_size=dake_cfg.get("window_size", 3),
        similarity_threshold=args.threshold or semantic_cfg.get("similarity_threshold", 0.90),
        device=args.device or mobile_cfg.get("device"),
        batch_size=args.batch_size or mobile_cfg.get("batch_size", 64),
    )


_PIPELINE_BUILDERS = {
    "pipeline_a": (build_pipeline_a, "configs/pipeline_a.yaml"),
    "pipeline_b": (build_pipeline_b, "configs/pipeline_b.yaml"),
    "pipeline_c": (build_pipeline_c, "configs/pipeline_c.yaml"),
    "pipeline_d": (build_pipeline_d, "configs/pipeline_d.yaml"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Keyframe Extractor Benchmark — AI Challenge 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chạy Pipeline C (đề xuất):
  python cli.py --pipeline pipeline_c --video_dir dataset/raw_video

  # Chạy tất cả pipeline:
  python cli.py --pipeline all --video_dir dataset/raw_video

  # Override tham số:
  python cli.py --pipeline pipeline_c --threshold 0.85 --candidate_ratio 0.05
        """,
    )

    # Pipeline selection
    parser.add_argument(
        "--pipeline",
        type=str,
        default="pipeline_c",
        choices=["pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d", "all"],
        help="Pipeline để chạy. 'all' sẽ chạy toàn bộ 4 pipeline lần lượt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn đến file YAML config (override mặc định).",
    )

    # I/O paths
    parser.add_argument(
        "--video_dir",
        type=str,
        default="dataset/raw_video",
        help="Thư mục chứa video .mp4.",
    )
    parser.add_argument(
        "--shots_dir",
        type=str,
        default="dataset/shots",
        help="Thư mục chứa shots.json của từng video.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark",
        help="Thư mục gốc lưu kết quả benchmark.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Thư mục chứa Ground Truth ảnh (.png/.jpg) của từng video để đánh giá.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        help="Giới hạn tên video cụ thể, cách nhau bởi dấu phẩy (vd: K01_V001.mp4,K01_V002.mp4). Không truyền = chạy toàn bộ video_dir.",
    )

    # Model overrides
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path đến BEiT-3 checkpoint .pth (override config).")
    parser.add_argument("--spm_path", type=str, default=None,
                        help="Path đến BEiT-3 .spm tokenizer (override config).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' hoặc 'cpu' (override config).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size cho encoder (override config).")

    # Algorithm params
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cosine similarity threshold [0-1] (override config).")
    parser.add_argument("--candidate_ratio", type=float, default=None,
                        help="DAKE candidate ratio [0.01-0.20] (override config).")

    args = parser.parse_args()

    # Validate paths
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"[CLI] ERROR: video_dir không tồn tại: {video_dir}")
        sys.exit(1)

    video_paths = sorted(video_dir.rglob("*.mp4"))
    if args.videos:
        wanted = set(args.videos.split(","))
        video_paths = [p for p in video_paths if p.name in wanted]
    if not video_paths:
        print(f"[CLI] ERROR: Không tìm thấy .mp4 trong {video_dir}")
        sys.exit(1)

    print(f"[CLI] Found {len(video_paths)} video(s) in {video_dir}")

    shots_dir = Path(args.shots_dir)
    output_dir = Path(args.output_dir)

    # Xác định danh sách pipeline cần chạy
    if args.pipeline == "all":
        pipeline_names = list(_PIPELINE_BUILDERS.keys())
    else:
        pipeline_names = [args.pipeline]

    # Import runner
    from src.core.runner import KeyframeBenchmarkRunner
    runner = KeyframeBenchmarkRunner(output_dir=output_dir)

    # Chạy từng pipeline
    for pipeline_name in pipeline_names:
        builder_fn, default_config = _PIPELINE_BUILDERS[pipeline_name]

        # Load config
        config_path = args.config or default_config
        if Path(config_path).exists():
            cfg = load_config(config_path).get("keyframe", {})
        else:
            print(f"[CLI] WARNING: Config không tìm thấy: {config_path}. Dùng giá trị mặc định.")
            cfg = {}

        extractor = builder_fn(cfg, args)
        gt_dir = Path(args.gt_dir) if args.gt_dir else None
        runner.run(extractor, video_paths, shots_dir, gt_dir=gt_dir)

    print(f"\n[CLI] Benchmark hoàn thành. Kết quả tại: {output_dir.resolve()}")
    print(f"[CLI] Summary CSV: {(output_dir / 'benchmark_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
