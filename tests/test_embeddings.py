"""Testes de labdados_core.embeddings (sem rede: cliente OpenAI mockado)."""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from labdados_core.embeddings import (
    EmbeddingConfig,
    build_chunks,
    build_result_zip,
    chunk_text,
    embed,
)


def test_chunk_texto_curto_e_vazio():
    assert chunk_text("  oi  ") == ["oi"]
    assert chunk_text("   ") == []


def test_chunk_respeita_tamanho_e_sobreposicao():
    frases = " ".join(f"Frase número {i} do documento." for i in range(200))
    chunks = chunk_text(frases, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # corta em fim de frase, não no meio da palavra
    assert all(c.endswith(".") for c in chunks[:-1])
    # todo o texto aparece em algum trecho
    for i in range(200):
        assert any(f"Frase número {i} " in c for c in chunks)


def test_chunk_parametros_invalidos():
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=10, overlap=10)


def test_build_chunks_csv_vira_um_documento_por_linha():
    csv = "id,ementa\n1,Ação de cobrança.\n2,Recurso não conhecido.\n".encode()
    chunks = build_chunks([("pasta/acordaos.csv", csv), ("nota.txt", b"Texto solto.")], csv_text_column="ementa")
    assert [(c["arquivo"], c["texto"]) for c in chunks] == [
        ("acordaos.csv", "Ação de cobrança."),
        ("acordaos.csv", "Recurso não conhecido."),
        ("nota.txt", "Texto solto."),
    ]
    assert len({c["doc_id"] for c in chunks}) == 3
    assert all(c["chunk"] == 0 for c in chunks)


def test_embed_openai_em_lotes_e_na_ordem(monkeypatch):
    client = MagicMock()

    def create(model, input):
        # devolve fora de ordem para garantir que o core reordena por index
        data = [SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))

    client.embeddings.create.side_effect = create
    fake_openai = MagicMock(return_value=client)
    monkeypatch.setattr("openai.OpenAI", fake_openai)

    vecs = embed(["a", "bb", "ccc"], EmbeddingConfig(model="m", base_url="http://x/v1", api_key="k", batch_size=2))

    assert vecs == [[1.0], [2.0], [3.0]]
    assert client.embeddings.create.call_count == 2
    assert fake_openai.call_args.kwargs["base_url"] == "http://x/v1"


def test_embed_vazio_nao_chama_api():
    assert embed([], EmbeddingConfig(model="m")) == []


def test_result_zip_roundtrip():
    chunks = build_chunks([("a.txt", b"Primeiro."), ("b.txt", b"Segundo.")])
    blob = build_result_zip(chunks, [[0.1, 0.2], [0.3, 0.4]], {"modelo": "m", "max_chars": 2000})
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert set(zf.namelist()) == {"embeddings.parquet", "chunks.csv", "_reproducibilidade/parametros.json"}
        df = pd.read_parquet(io.BytesIO(zf.read("embeddings.parquet")))
        params = json.loads(zf.read("_reproducibilidade/parametros.json"))
        csv = zf.read("chunks.csv").decode()
    assert list(df.columns) == ["arquivo", "doc_id", "chunk", "texto", "embedding"]
    assert [list(v) for v in df["embedding"]] == [[0.1, 0.2], [0.3, 0.4]]
    assert params == {"modelo": "m", "max_chars": 2000, "dimensoes": 2, "trechos": 2}
    assert "embedding" not in csv.splitlines()[0]


def test_result_zip_valida_contagem():
    chunks = build_chunks([("a.txt", b"x")])
    with pytest.raises(ValueError):
        build_result_zip(chunks, [], {})
