"""彙整 artifacts/*/metadata.json 產出模型比較表：python -m stockta.ml.compare

輸出 Markdown 到 docs/model_comparison.md，可直接進專題報告。
"""

from __future__ import annotations

import json
from pathlib import Path

from stockta.config import ARTIFACTS_DIR, BACKEND_ROOT, LABEL_CLASSES

DOCS_DIR = BACKEND_ROOT.parent / "docs"


def build_table(artifacts_dir: Path = ARTIFACTS_DIR) -> str:
    rows = []
    for meta_path in sorted(artifacts_dir.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        test = meta.get("metrics", {}).get("test", {})
        val = meta.get("metrics", {}).get("val", {})
        rows.append(
            {
                "模型": meta["model_name"],
                "驗證 Macro AUC": _fmt(val.get("macro_auc")),
                "測試 Macro AUC": _fmt(test.get("macro_auc")),
                "測試 Accuracy": _fmt(test.get("accuracy")),
                "測試 Macro F1": _fmt(test.get("macro_f1")),
                "多數類基線 Acc": _fmt(test.get("majority_baseline_accuracy")),
                "訓練耗時(s)": meta.get("metrics", {}).get("train_seconds", "—"),
                "訓練日": meta.get("trained_at", "")[:10],
            }
        )
    if not rows:
        return "（尚無任何 artifact，先執行 python -m stockta.ml.train）\n"

    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def main() -> None:
    table = build_table()
    doc = (
        "# 模型比較報告\n\n"
        f"標籤三分類（{ ' / '.join(LABEL_CLASSES) }），時間序列切分，"
        "指標以測試集為準；多數類基線 Accuracy 是必須超越的對手。\n\n"
        f"{table}\n"
        "> 由 `python -m stockta.ml.compare` 自動產生，數據來源為 "
        "`backend/artifacts/*/metadata.json`。\n"
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "model_comparison.md"
    out.write_text(doc, encoding="utf-8")
    print(table)
    print(f"已寫入 {out}")


if __name__ == "__main__":
    main()
