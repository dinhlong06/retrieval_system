"""
analyze.py -- gộp output/qwen.json + output/prod.json thành báo cáo so sánh.

Qwen đọc cả khung hình và trả về văn bản tự do, baseline (output_hybrid.json
có sẵn của ../OCR) trả về từng dòng có box, nên không so được theo từng box.
Thước đo dùng ở đây là **recall theo dòng**: mỗi dòng baseline đọc được có
xuất hiện trong văn bản Qwen không, sau khi chuẩn hóa cả hai bên về ascii
(bỏ dấu, bỏ ký tự không phải chữ/số, hạ chữ thường) -- pipeline search dùng
ascii mặc định nên dấu đúng/sai không ảnh hưởng đến việc tìm được câu đó hay
không, chỉ nội dung chữ mới quan trọng.

Recall cao nghĩa là Qwen bao được nội dung baseline đọc ra -- nó KHÔNG nói
Qwen đúng, vì cả hai đều có thể sai giống nhau và Qwen còn có thể bịa thêm.
Phần cuối in ra các frame lệch nhiều nhất để chấm bằng mắt.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def norm(s: str) -> str:
    s = s.replace("Đ", "D").replace("đ", "d")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return _NON_ALNUM_RE.sub("", s.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--show", type=int, default=20, help="số frame lệch nhiều nhất cần in")
    ap.add_argument("--report", default="output/report.txt")
    args = ap.parse_args()

    out = Path(args.output_dir)
    qwen = {r["frame_id"]: r for r in json.loads((out / "qwen.json").read_text(encoding="utf-8"))}
    prod = {r["frame_id"]: r for r in json.loads((out / "prod.json").read_text(encoding="utf-8"))}
    ids = sorted(set(qwen) & set(prod))

    lines, per_frame = [], []
    n_q_chars = 0
    hit = total = 0
    for fid in ids:
        q = qwen[fid]["qwen"]
        base = [t["text"] for t in prod[fid]["prod"]]
        qn = norm(q)
        # Chỉ tính dòng đủ dài: dòng 1-2 ký tự trùng ngẫu nhiên rất dễ.
        checked = [x for x in base if len(norm(x)) >= 4]
        found = [x for x in checked if norm(x) in qn]
        hit += len(found)
        total += len(checked)
        n_q_chars += len(q)
        miss = [x for x in checked if x not in found]
        per_frame.append((len(miss), fid, q, base, miss))

    lines.append(f"frames: {len(ids)}")
    lines.append(f"recall dòng (ascii, dấu không tính): {hit}/{total} = {100*hit/max(total,1):.1f}%")
    lines.append(f"độ dài trung bình text Qwen mỗi frame: {n_q_chars/max(len(ids),1):.0f} ký tự")
    lines.append("\nRecall KHÔNG phải điểm chính xác: nó chỉ đo độ phủ so với baseline hiện tại,")
    lines.append("không bắt được trường hợp Qwen bịa thêm text không có trong ảnh.")

    lines.append(f"\n--- {args.show} frame Qwen bỏ sót nhiều dòng nhất ---")
    for _, fid, q, base, miss in sorted(per_frame, reverse=True)[: args.show]:
        lines.append(f"\n[{fid}]  {qwen[fid]['path']}")
        lines.append(f"  qwen : {q!r}")
        lines.append(f"  prod : {base}")
        lines.append(f"  thiếu: {miss}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"\n-> {args.report}")


if __name__ == "__main__":
    main()
