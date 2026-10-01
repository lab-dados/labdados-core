"""Tests para o pipeline de estruturação.

Mockam o openai SDK no nível da fábrica de cliente para que os tests
não dependam de chave nem de rede.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from labdados_core.estruturacao import (
    LlmConfig,
    build_messages,
    call_llm,
    estruturar,
    read_document,
)
from labdados_core.estruturacao.prompts import DEFAULT_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_build_messages_default_user_position():
    msgs = build_messages("contexto", "texto do doc", {"type": "object"})
    assert msgs[0] == {"role": "system", "content": "contexto"}
    assert msgs[1]["role"] == "user"
    assert "Texto:\ntexto do doc" in msgs[1]["content"]
    assert "Schema esperado:" in msgs[1]["content"]


def test_build_messages_system_position_legacy_sdk():
    msgs = build_messages("ctx", "txt", {"type": "object"}, schema_position="system")
    assert "JSON válido seguindo este schema" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "txt"}


def test_build_messages_default_system_when_blank():
    msgs = build_messages("", "txt", None)
    assert msgs[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert msgs[1]["content"] == "txt"


def test_build_messages_no_schema_means_user_text_only():
    msgs = build_messages("ctx", "texto puro", None)
    assert msgs[1] == {"role": "user", "content": "texto puro"}


# ---------------------------------------------------------------------------
# _llm
# ---------------------------------------------------------------------------


def _fake_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _patch_client(monkeypatch, response_content: str = '{"ok": true}'):
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(response_content)
    monkeypatch.setattr(
        "labdados_core.estruturacao._llm._make_client", lambda config: client
    )
    return client


def test_call_llm_basic_json(monkeypatch):
    client = _patch_client(monkeypatch, '{"x": 1}')
    config = LlmConfig(model="gpt-4o-mini", api_key="sk-test")
    result = call_llm([{"role": "user", "content": "hi"}], config=config)
    assert result == {"x": 1}
    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-4o-mini"
    assert create_kwargs["response_format"] == {"type": "json_object"}


def test_call_llm_with_schema_uses_json_schema(monkeypatch):
    client = _patch_client(monkeypatch, '{"a": "b"}')
    config = LlmConfig(model="gpt-4o-mini")
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    call_llm([{"role": "user", "content": "hi"}], config=config, schema=schema)
    rf = client.chat.completions.create.call_args.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    # strict mode exige additionalProperties:false e required completo (ver _strictify_schema)
    sent = rf["json_schema"]["schema"]
    assert sent["additionalProperties"] is False
    assert sent["required"] == ["a"]
    assert sent["properties"] == {"a": {"type": "string"}}


def test_strictify_schema_nested_and_arrays():
    """additionalProperties:false + required completo em objetos aninhados e items."""
    from labdados_core.estruturacao._llm import _strictify_schema

    schema = {
        "type": "object",
        "properties": {
            "autor": {"type": "string"},
            "enderecos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cidade": {"type": "string"},
                        "uf": {"type": "string"},
                    },
                },
            },
        },
    }
    out = _strictify_schema(schema)

    assert out["additionalProperties"] is False
    assert sorted(out["required"]) == ["autor", "enderecos"]
    item = out["properties"]["enderecos"]["items"]
    assert item["additionalProperties"] is False
    assert sorted(item["required"]) == ["cidade", "uf"]
    # Não muta a entrada original.
    assert "additionalProperties" not in schema
    assert "required" not in schema


def test_strictify_schema_ignores_non_dict():
    from labdados_core.estruturacao._llm import _strictify_schema

    assert _strictify_schema(None) is None
    assert _strictify_schema("x") == "x"


def test_call_llm_openai_uses_max_completion_tokens(monkeypatch):
    client = _patch_client(monkeypatch)
    config = LlmConfig(
        provider="azure_openai",
        model="analise",
        base_url="https://x.openai.azure.com/",
        api_version="2024-12-01-preview",
        max_tokens=1234,
    )
    call_llm([{"role": "user", "content": "hi"}], config=config)
    kw = client.chat.completions.create.call_args.kwargs
    assert kw["max_completion_tokens"] == 1234
    assert "max_tokens" not in kw
    assert kw["temperature"] == 0.0


def test_call_llm_openai_compat_keeps_max_tokens(monkeypatch):
    client = _patch_client(monkeypatch)
    config = LlmConfig(provider="openai_compat", model="llama3.1", base_url="http://x/v1", max_tokens=99)
    call_llm([{"role": "user", "content": "hi"}], config=config)
    kw = client.chat.completions.create.call_args.kwargs
    assert kw["max_tokens"] == 99
    assert "max_completion_tokens" not in kw


def test_call_llm_reasoning_model_omits_temperature(monkeypatch):
    client = _patch_client(monkeypatch)
    config = LlmConfig(model="gpt-5.6-luna", temperature=None, reasoning_effort="low")
    call_llm([{"role": "user", "content": "hi"}], config=config)
    kw = client.chat.completions.create.call_args.kwargs
    assert "temperature" not in kw
    assert kw["reasoning_effort"] == "low"


def test_to_dataframeit_kwargs_reasoning_model():
    kwargs = to_dataframeit_kwargs(
        LlmConfig(model="gpt-5.6-luna", api_key="sk-x", temperature=None, reasoning_effort="low")
    )
    # None explícito sobrescreve o temperature=0 default do DataFrameIt
    assert kwargs["model_kwargs"]["temperature"] is None
    assert kwargs["model_kwargs"]["reasoning_effort"] == "low"


def test_call_llm_invalid_json_returns_raw(monkeypatch):
    _patch_client(monkeypatch, "isso não é json {")
    config = LlmConfig(model="gpt-4o-mini")
    result = call_llm([{"role": "user", "content": "hi"}], config=config)
    assert result == {"_raw_response": "isso não é json {"}


def test_call_llm_streaming(monkeypatch):
    client = MagicMock()

    def make_event(piece):
        delta = MagicMock()
        delta.content = piece
        choice = MagicMock()
        choice.delta = delta
        ev = MagicMock()
        ev.choices = [choice]
        return ev

    client.chat.completions.create.return_value = iter(
        [make_event('{"ok"'), make_event(": "), make_event("true}")]
    )
    monkeypatch.setattr(
        "labdados_core.estruturacao._llm._make_client", lambda config: client
    )

    config = LlmConfig(model="m", stream=True)
    result = call_llm([{"role": "user", "content": "hi"}], config=config)
    assert result == {"ok": True}
    assert client.chat.completions.create.call_args.kwargs["stream"] is True


def test_llm_config_azure_requires_endpoint():
    from labdados_core.estruturacao._llm import _make_client

    config = LlmConfig(model="dep", provider="azure_openai", api_version="2024-12-01")
    with pytest.raises(ValueError, match="base_url"):
        _make_client(config)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def test_read_txt():
    docs = read_document(b"linha 1\nlinha 2", "doc.txt")
    assert docs == [("doc", "linha 1\nlinha 2")]


def test_read_md_treats_as_plain_text():
    docs = read_document(b"# titulo", "notas.md")
    assert docs[0][1] == "# titulo"


def test_read_csv_per_row_with_text_column():
    csv_bytes = b"id,texto\n1,primeiro\n2,segundo\n"
    docs = read_document(csv_bytes, "data.csv", csv_text_column="texto")
    assert [d[1] for d in docs] == ["primeiro", "segundo"]
    assert docs[0][0] == "data__row0001"


def test_read_csv_concat_when_no_text_column():
    csv_bytes = b"id,a,b\n1,foo,bar\n"
    [(_, text)] = read_document(csv_bytes, "data.csv")
    assert "id: 1" in text
    assert "a: foo" in text
    assert "b: bar" in text


def test_read_csv_text_column_case_insensitive():
    csv_bytes = b"ID,Texto\n1,hello\n"
    [(_, text)] = read_document(csv_bytes, "x.csv", csv_text_column="texto")
    assert text == "hello"


def test_read_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Extensão não suportada"):
        read_document(b"...", "foo.pdf")


def test_read_xlsx_per_row():
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["id", "texto"])
    ws.append([1, "alpha"])
    ws.append([2, "beta"])
    buf = io.BytesIO()
    wb.save(buf)

    docs = read_document(buf.getvalue(), "data.xlsx", csv_text_column="texto")
    assert [d[1] for d in docs] == ["alpha", "beta"]


def test_read_docx_extracts_paragraphs():
    """Build a minimal valid DOCX inline (zip + word/document.xml)."""
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>Primeiro paragrafo.</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Segundo paragrafo.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    [(_, text)] = read_document(buf.getvalue(), "doc.docx")
    assert text == "Primeiro paragrafo.\nSegundo paragrafo."


# ---------------------------------------------------------------------------
# schema_utils (JSON Schema -> Pydantic)
# ---------------------------------------------------------------------------


from pydantic import BaseModel  # noqa: E402

from labdados_core.estruturacao import UnsupportedSchema, ensure_pydantic_model  # noqa: E402


class _Sample(BaseModel):
    a: str
    b: int | None = None


def test_ensure_pydantic_model_passes_through_basemodel():
    assert ensure_pydantic_model(_Sample) is _Sample


def test_ensure_pydantic_model_converts_simple_dict():
    schema = {
        "type": "object",
        "properties": {
            "nome": {"type": "string", "description": "nome do autor"},
            "idade": {"type": "integer"},
        },
        "required": ["nome"],
    }
    Model = ensure_pydantic_model(schema)
    assert issubclass(Model, BaseModel)
    fields = Model.model_fields
    assert fields["nome"].is_required()
    assert fields["nome"].description == "nome do autor"
    assert not fields["idade"].is_required()


def test_ensure_pydantic_model_handles_array_and_nested():
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
            "endereco": {
                "type": "object",
                "properties": {
                    "cidade": {"type": "string"},
                    "uf": {"type": "string"},
                },
                "required": ["cidade"],
            },
        },
    }
    Model = ensure_pydantic_model(schema)
    instance = Model(tags=["a", "b"], endereco={"cidade": "São Paulo", "uf": "SP"})
    assert instance.tags == ["a", "b"]
    assert instance.endereco.cidade == "São Paulo"


def test_ensure_pydantic_model_rejects_unsupported_construct():
    with pytest.raises(UnsupportedSchema, match="anyOf"):
        ensure_pydantic_model(
            {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}]}}}
        )


def test_ensure_pydantic_model_rejects_non_object_root():
    with pytest.raises(UnsupportedSchema, match="type=object"):
        ensure_pydantic_model({"type": "string"})


def test_ensure_pydantic_model_handles_enum_as_literal():
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["a", "b", "c"]}},
        "required": ["status"],
    }
    Model = ensure_pydantic_model(schema)
    from pydantic import ValidationError

    Model(status="a")  # ok
    with pytest.raises(ValidationError):
        Model(status="d")


def test_ensure_pydantic_model_rejects_non_dict_non_class():
    with pytest.raises(TypeError):
        ensure_pydantic_model("not a schema")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# to_dataframeit_kwargs
# ---------------------------------------------------------------------------


from labdados_core.estruturacao import to_dataframeit_kwargs  # noqa: E402


def test_to_dataframeit_kwargs_openai_basic():
    kwargs = to_dataframeit_kwargs(LlmConfig(model="gpt-4o-mini", api_key="sk-x"))
    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["api_key"] == "sk-x"
    assert kwargs["model_kwargs"]["temperature"] == 0.0


def test_to_dataframeit_kwargs_openai_compat_uses_base_url():
    kwargs = to_dataframeit_kwargs(
        LlmConfig(
            provider="openai_compat",
            model="llama3.1",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    )
    assert kwargs["provider"] == "openai"  # langchain só conhece "openai"
    assert kwargs["model_kwargs"]["base_url"] == "http://localhost:11434/v1"
    assert kwargs["api_key"] == "ollama"


def test_to_dataframeit_kwargs_azure_openai():
    kwargs = to_dataframeit_kwargs(
        LlmConfig(
            provider="azure_openai",
            model="gpt-4.1-mini",
            api_key="azure-key",
            base_url="https://x.openai.azure.com/",
            api_version="2024-12-01-preview",
        )
    )
    assert kwargs["provider"] == "azure_openai"
    assert kwargs["model_kwargs"]["azure_endpoint"] == "https://x.openai.azure.com/"
    assert kwargs["model_kwargs"]["api_version"] == "2024-12-01-preview"


def test_to_dataframeit_kwargs_azure_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        to_dataframeit_kwargs(
            LlmConfig(provider="azure_openai", model="m", api_version="2024-12-01")
        )


def test_to_dataframeit_kwargs_azure_requires_api_version():
    with pytest.raises(ValueError, match="api_version"):
        to_dataframeit_kwargs(
            LlmConfig(provider="azure_openai", model="m", base_url="https://x")
        )


# ---------------------------------------------------------------------------
# pipeline (estruturar) — mocka dataframeit
# ---------------------------------------------------------------------------


def _patch_dataframeit(monkeypatch, side_effect=None, error_per_text=None):
    """Substitui dataframeit() por uma fake que devolve um DataFrame plausível.

    Se ``side_effect`` for um dict {texto: dict_de_campos}, cada linha do
    DataFrame de entrada vira uma linha no DataFrame de saída com esses
    campos. Se ``error_per_text`` for um dict {texto: msg}, popula
    ``_error_details`` para essas linhas.
    """

    calls: list[Any] = []

    def fake(df, *, questions, prompt, text_column, **kwargs):
        calls.append(
            {
                "df": df.copy(),
                "questions": questions,
                "prompt": prompt,
                "text_column": text_column,
                "kwargs": kwargs,
            }
        )
        out = df.copy()
        # popula campos do model com o valor do mock por texto
        field_names = list(questions.model_fields.keys())
        for fn in field_names:
            out[fn] = None
        if side_effect:
            for i, txt in enumerate(out[text_column]):
                if txt in side_effect:
                    for fn, val in side_effect[txt].items():
                        out.at[out.index[i], fn] = val
        if error_per_text:
            out["_error_details"] = None
            for i, txt in enumerate(out[text_column]):
                if txt in error_per_text:
                    out.at[out.index[i], "_error_details"] = error_per_text[txt]
        return out

    monkeypatch.setattr("labdados_core.estruturacao.pipeline.dataframeit", fake, raising=False)
    # dataframeit é importado dentro de _run_dataframeit. Para garantir que o
    # patch pegue, sobrescreve no módulo dataframeit também.
    import dataframeit as _df_mod

    monkeypatch.setattr(_df_mod, "dataframeit", fake)
    return calls


def test_estruturar_single_string(monkeypatch):
    calls = _patch_dataframeit(monkeypatch, side_effect={"texto": {"a": "ok"}})
    result = estruturar("texto", schema=_Sample, llm_config=LlmConfig(model="m"))
    assert len(result) == 1
    assert result[0]["_doc_id"] == "doc_1"
    assert result[0]["a"] == "ok"
    assert len(calls) == 1


def test_estruturar_list_of_strings_assigns_ids(monkeypatch):
    _patch_dataframeit(monkeypatch, side_effect={"a": {"a": "x"}, "b": {"a": "y"}})
    result = estruturar(["a", "b"], schema=_Sample, llm_config=LlmConfig(model="m"))
    assert [r["_doc_id"] for r in result] == ["doc_1", "doc_2"]


def test_estruturar_preserves_explicit_doc_ids(monkeypatch):
    _patch_dataframeit(
        monkeypatch, side_effect={"texto a": {"a": "1"}, "texto b": {"a": "2"}}
    )
    result = estruturar(
        [("alfa", "texto a"), ("beta", "texto b")],
        schema=_Sample,
        llm_config=LlmConfig(model="m"),
    )
    assert [r["_doc_id"] for r in result] == ["alfa", "beta"]


def test_estruturar_skips_empty_text(monkeypatch):
    calls = _patch_dataframeit(monkeypatch, side_effect={})
    result = estruturar(["", "  "], schema=_Sample, llm_config=LlmConfig(model="m"))
    assert all(r["_error"] == "documento vazio" for r in result)
    assert calls == []  # dataframeit nem foi chamado (só docs vazios)


def test_estruturar_mixed_empty_and_valid(monkeypatch):
    _patch_dataframeit(monkeypatch, side_effect={"bom": {"a": "ok"}})
    result = estruturar(["", "bom"], schema=_Sample, llm_config=LlmConfig(model="m"))
    assert result[0]["_error"] == "documento vazio"
    assert result[1]["a"] == "ok"


def test_estruturar_captures_dataframeit_exception(monkeypatch):
    def fake(df, *, questions, prompt, text_column, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "labdados_core.estruturacao.pipeline.dataframeit", fake, raising=False
    )
    import dataframeit as _df_mod

    monkeypatch.setattr(_df_mod, "dataframeit", fake)

    result = estruturar(["a", "b"], schema=_Sample, llm_config=LlmConfig(model="m"))
    assert all(r["_error"] == "boom" for r in result)


def test_estruturar_propagates_per_row_error_from_df(monkeypatch):
    _patch_dataframeit(
        monkeypatch,
        side_effect={"bom": {"a": "ok"}},
        error_per_text={"falha": "rate limit"},
    )
    result = estruturar(
        ["bom", "falha"], schema=_Sample, llm_config=LlmConfig(model="m")
    )
    assert result[0]["a"] == "ok"
    assert result[1]["_error"] == "rate limit"


def test_estruturar_accepts_dict_schema(monkeypatch):
    """JSON Schema dict é aceito (convertido pra Pydantic internamente)."""
    schema_dict = {
        "type": "object",
        "properties": {"resumo": {"type": "string"}},
        "required": ["resumo"],
    }
    calls = _patch_dataframeit(monkeypatch, side_effect={"texto": {"resumo": "ok"}})
    result = estruturar(
        "texto", schema=schema_dict, llm_config=LlmConfig(model="m")
    )
    assert result[0]["resumo"] == "ok"
    # Verifica que o dataframeit recebeu uma classe Pydantic, não o dict.
    assert issubclass(calls[0]["questions"], BaseModel)


def test_estruturar_passes_system_prompt_in_template(monkeypatch):
    calls = _patch_dataframeit(monkeypatch, side_effect={"t": {"a": "x"}})
    estruturar(
        "t",
        schema=_Sample,
        system_prompt="extraia decisões judiciais",
        llm_config=LlmConfig(model="m"),
    )
    sent_prompt = calls[0]["prompt"]
    assert "extraia decisões judiciais" in sent_prompt
    assert "{texto}" in sent_prompt  # placeholder preservado pra dataframeit


def test_estruturar_passes_llm_config_through(monkeypatch):
    calls = _patch_dataframeit(monkeypatch, side_effect={"t": {"a": "x"}})
    estruturar(
        "t",
        schema=_Sample,
        llm_config=LlmConfig(
            provider="openai_compat",
            model="llama3.1",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
    )
    df_kwargs = calls[0]["kwargs"]
    assert df_kwargs["provider"] == "openai"
    assert df_kwargs["model"] == "llama3.1"
    assert df_kwargs["model_kwargs"]["base_url"] == "http://localhost:11434/v1"
