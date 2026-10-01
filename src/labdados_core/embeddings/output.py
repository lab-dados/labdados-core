"""Formato de saída dos embeddings — igual no serviço e no SDK."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from labdados_core.embeddings.pipeline import Chunk


def _frame(chunks: list[Chunk], vectors: list[list[float]] | None):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas não instalado. Adicione `labdados-core[embeddings]` às deps.") from exc

    df = pd.DataFrame(chunks, columns=["arquivo", "doc_id", "chunk", "texto"])
    if vectors is not None:
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vetores para {len(chunks)} trechos")
        df["embedding"] = vectors
    return df


def to_parquet_bytes(chunks: list[Chunk], vectors: list[list[float]]) -> bytes:
    """``embeddings.parquet`` com colunas arquivo, doc_id, chunk, texto, embedding."""
    buf = io.BytesIO()
    _frame(chunks, vectors).to_parquet(buf, index=False)
    return buf.getvalue()


def build_result_zip(chunks: list[Chunk], vectors: list[list[float]], parametros: dict[str, Any]) -> bytes:
    """Empacota o resultado.

    - ``embeddings.parquet`` — um trecho por linha, com o vetor
      (``pandas.read_parquet`` / ``arrow::read_parquet``);
    - ``chunks.csv`` — os mesmos trechos sem os vetores, para inspeção rápida;
    - ``_reproducibilidade/parametros.json`` — modelo, tamanho do trecho,
      sobreposição, dimensão e contagens.
    """
    dims = len(vectors[0]) if vectors else 0
    params = {**parametros, "dimensoes": dims, "trechos": len(chunks)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("embeddings.parquet", to_parquet_bytes(chunks, vectors))
        zf.writestr("chunks.csv", _frame(chunks, None).to_csv(index=False))
        zf.writestr(
            "_reproducibilidade/parametros.json",
            json.dumps(params, indent=2, ensure_ascii=False),
        )
    return buf.getvalue()
