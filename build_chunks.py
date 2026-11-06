"""정제 정책 원문을 정책 항목의 경계를 최대한 유지하며 검색 청크로 변환한다."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def split_oversized(text: str, limit: int = 850) -> list[str]:
    """긴 문단을 문장 종결부 위주로 나눈다."""
    if len(text) <= limit:
        return [text]
    result: list[str] = []
    left = text.strip()
    while len(left) > limit:
        pos = max(left.rfind(". ", 0, limit), left.rfind("。", 0, limit), left.rfind("; ", 0, limit))
        if pos < limit // 2:
            pos = max(left.rfind(" ", 0, limit), left.rfind("\n", 0, limit))
        if pos < limit // 2:
            pos = limit
        else:
            pos += 1
        result.append(left[:pos].strip())
        left = left[pos:].strip()
    if left:
        result.append(left)
    return result


def make_chunks(text: str, max_chars: int = 1100, overlap: int = 90) -> list[str]:
    """원문 줄과 항목 경계를 보존하면서 긴 본문을 분할한다."""
    normalized = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return []

    # 가능한 한 정책 번호와 해당 설명을 같은 청크에 남긴다.
    pieces: list[str] = []
    buffer: list[str] = []
    for line in lines:
        for part in split_oversized(line, max_chars - 80):
            candidate = "\n".join(buffer + [part])
            if buffer and len(candidate) > max_chars:
                pieces.append("\n".join(buffer))
                buffer = []
            buffer.append(part)
    if buffer:
        pieces.append("\n".join(buffer))

    # 긴 문서의 연속 문맥만 소량 중복하고, 날짜/조항 번호는 삭제하지 않는다.
    if len(pieces) <= 1 or overlap <= 0:
        return pieces
    result = [pieces[0]]
    for i in range(1, len(pieces)):
        result.append(pieces[i - 1][-overlap:] + "\n" + pieces[i])
    return result


def build_chunks(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"ID", "TITLE_CLEAN", "REG_DATE", "PUBLISHER", "ABSTRACT_CLEAN", "POLICY_CONTEXT_CLEAN"}
    if missing := required - set(dataframe.columns):
        raise ValueError(f"정제 데이터의 필수 컬럼이 없습니다: {sorted(missing)}")
    rows = []
    for row in dataframe.fillna("").to_dict("records"):
        doc_id = str(row["ID"]).strip()
        title = str(row["TITLE_CLEAN"]).strip()
        for column in ("POLICY_CONTEXT_CLEAN", "ABSTRACT_CLEAN"):
            for idx, content in enumerate(make_chunks(row[column]), 1):
                rows.append({
                    "chunk_id": f"{doc_id}:{'context' if column.startswith('POLICY') else 'abstract'}:{idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "reg_date": str(row["REG_DATE"]),
                    "publisher": str(row["PUBLISHER"]),
                    "source_column": column,
                    "section": content.split("\n", 1)[0][:100],
                    "text": content,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "policies.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "chunks.csv")
    args = parser.parse_args()
    docs = pd.read_csv(args.source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    chunks = build_chunks(docs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"문서 {len(docs)}개 → 텍스트 청크 {len(chunks)}개: {args.output}")


if __name__ == "__main__":
    main()
