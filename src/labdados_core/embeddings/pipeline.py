"""Leitura de documentos em trechos e cálculo dos vetores."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, TypedDict

from labdados_core.embeddings.chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, chunk_text
from labdados_core.estruturacao.readers import read_document

Provider = Literal["openai", "sentence_transformers"]

# Multilíngue, pequeno (~120 MB, 384 dims) e sem prefixo de instrução — roda
# em CPU de notebook.
DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Chunk(TypedDict):
    """Um trecho de documento. ``doc_id`` vem do reader (linha do CSV, nome do arquivo…)."""

    arquivo: str
    doc_id: str
    chunk: int
    texto: str


@dataclass
class EmbeddingConfig:
    """Como calcular os vetores.

    - ``provider="openai"``: qualquer endpoint OpenAI-compatível — Azure AI
      Foundry (``https://<recurso>.openai.azure.com/openai/v1/``), OpenAI,
      Ollama (``http://localhost:11434/v1``). ``model`` é o nome do
      deployment/modelo.
    - ``provider="sentence_transformers"``: modelo local do Hugging Face
      (requer o extra ``[embeddings-local]``).
    """

    model: str
    provider: Provider = "openai"
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = 32
    timeout: float = 120.0


def build_chunks(
    files: list[tuple[str, bytes]],
    *,
    csv_text_column: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Lê cada ``(nome_do_arquivo, bytes)`` e devolve os trechos na ordem.

    CSV/XLSX: cada linha vira um documento (``csv_text_column`` escolhe a
    coluna; vazio concatena todas), como na estruturação.
    """
    chunks: list[Chunk] = []
    for filename, content in files:
        arquivo = os.path.basename(filename)
        for doc_id, text in read_document(content, arquivo, csv_text_column=csv_text_column):
            for i, piece in enumerate(chunk_text(text, max_chars=max_chars, overlap=overlap)):
                chunks.append({"arquivo": arquivo, "doc_id": doc_id, "chunk": i, "texto": piece})
    return chunks


def _embed_openai(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai não instalado. Adicione `labdados-core[embeddings]` às deps.") from exc

    kwargs: dict = {"api_key": config.api_key or "unused", "timeout": config.timeout, "max_retries": 5}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    client = OpenAI(**kwargs)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), config.batch_size):
        batch = texts[i : i + config.batch_size]
        resp = client.embeddings.create(model=config.model, input=batch)
        # A API devolve um item por entrada com ``index``; ordena por garantia.
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return vectors


def _embed_sentence_transformers(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers não instalado. Adicione `labdados-core[embeddings-local]` às deps."
        ) from exc

    model = SentenceTransformer(config.model)
    return model.encode(texts, batch_size=config.batch_size, convert_to_numpy=True).tolist()


def embed(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    """Calcula um vetor por texto, na mesma ordem."""
    if not texts:
        return []
    if config.provider == "openai":
        return _embed_openai(texts, config)
    if config.provider == "sentence_transformers":
        return _embed_sentence_transformers(texts, config)
    raise ValueError(f"provider de embeddings desconhecido: {config.provider!r}")
