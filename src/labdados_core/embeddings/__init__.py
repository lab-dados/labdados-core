"""Embeddings de textos — compartilhado entre o serviço do escritório e o SDK.

Fica no core o que precisa sair idêntico nos dois lados:

- **leitura + chunking** (:func:`build_chunks`) — mesmos readers da
  estruturação (``.txt/.md/.docx/.csv/.xlsx``), mesmo corte por caracteres
  com sobreposição;
- **chamada ao modelo** (:func:`embed`) — qualquer endpoint OpenAI-compatível
  (Azure AI Foundry, OpenAI, Ollama…) ou ``sentence-transformers`` local;
- **formato de saída** (:func:`build_result_zip`) — ``embeddings.parquet``
  (arquivo, doc_id, chunk, texto, embedding), ``chunks.csv`` sem os vetores
  e ``_reproducibilidade/parametros.json``.

Extras: ``[embeddings]`` (cliente OpenAI + pandas/pyarrow) e
``[embeddings-local]`` (adiciona ``sentence-transformers``).
"""

from labdados_core.embeddings.chunking import chunk_text
from labdados_core.embeddings.output import build_result_zip, to_parquet_bytes
from labdados_core.embeddings.pipeline import (
    DEFAULT_LOCAL_MODEL,
    Chunk,
    EmbeddingConfig,
    build_chunks,
    embed,
)

__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "Chunk",
    "EmbeddingConfig",
    "build_chunks",
    "build_result_zip",
    "chunk_text",
    "embed",
    "to_parquet_bytes",
]
