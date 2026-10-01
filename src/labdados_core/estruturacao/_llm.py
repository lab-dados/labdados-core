"""Configuração de LLM e wrappers de chamada.

Todos os providers que aparecem no projeto são OpenAI-compatible no
nível da API:

- **OpenAI** direto — ``provider="openai"``.
- **Azure OpenAI** — ``provider="azure_openai"`` + ``base_url`` (azure
  endpoint), ``api_version``, ``model`` (= deployment name).
- **OpenAI-compat self-host** (vLLM em Container Apps, Ollama local,
  LM Studio, OpenRouter) — ``provider="openai_compat"`` + ``base_url``.

A função :func:`to_dataframeit_kwargs` mapeia :class:`LlmConfig` para os
kwargs aceitos por ``dataframeit()``, que por baixo usa
``langchain.init_chat_model`` — o ``base_url`` vai dentro de
``model_kwargs`` e ``init_chat_model`` repassa pra
``ChatOpenAI``/``AzureChatOpenAI``.

A função :func:`call_llm` mantém um cliente OpenAI direto para casos
em que se quer uma única chamada chat completion sem o overhead do
DataFrameIt (DataFrame + tqdm + threadpool). É o caminho usado por
``services/structuring/main.py`` quando processa um doc por vez.

Notas operacionais:

- ``stream=True`` é **necessário** para vLLM atrás do ingress do Azure
  Container Apps — gerações longas sem stream batem em "stream timeout"
  ~240s mesmo com ``timeout=None`` no cliente.
- Quando há ``schema`` válido, ``call_llm`` usa
  ``response_format={"type": "json_schema", ...}`` (structured outputs
  / guided decoding); senão, ``json_object``. ``estruturar`` (via
  DataFrameIt) usa Pydantic structured output, que é equivalente.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Provider = Literal["openai", "azure_openai", "openai_compat"]


@dataclass
class LlmConfig:
    """Configuração para uma chamada LLM.

    Atributos relevantes por provider:

    - ``openai``: ``model``, ``api_key``.
    - ``azure_openai``: ``model`` (= deployment name), ``api_key``,
      ``base_url`` (Azure endpoint), ``api_version``.
    - ``openai_compat``: ``model``, ``base_url``, ``api_key`` (pode ser
      ``"unused"``/``"EMPTY"`` para vLLM/Ollama).

    Modelos de raciocínio (família GPT-5, ex.: ``gpt-5.6-luna`` no Azure
    da FGV) só aceitam a ``temperature`` default — passe
    ``temperature=None`` para omitir o parâmetro. ``reasoning_effort``
    (``"minimal"``/``"low"``/``"medium"``/``"high"``) só é enviado se
    definido.
    """

    model: str
    provider: Provider = "openai"
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    temperature: float | None = 0.0
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    stream: bool = False
    timeout: float | None = 120.0
    extra: dict[str, Any] = field(default_factory=dict)


def _make_client(config: LlmConfig):
    """Instancia o cliente OpenAI/AzureOpenAI conforme o provider.

    Lazy import para que ``import labdados_core.estruturacao`` não force
    o openai SDK em quem não precisa (ex.: backend que importa só
    ``contracts``).
    """
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai não instalado. Adicione `labdados-core[estruturacao]` às deps."
        ) from exc

    if config.provider == "azure_openai":
        if not config.base_url:
            raise ValueError("azure_openai requer base_url (azure_endpoint)")
        if not config.api_version:
            raise ValueError("azure_openai requer api_version")
        return AzureOpenAI(
            azure_endpoint=config.base_url,
            api_key=config.api_key,
            api_version=config.api_version,
            timeout=config.timeout,
        )

    kwargs: dict[str, Any] = {"api_key": config.api_key or "unused", "timeout": config.timeout}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def _strictify_schema(schema: Any) -> Any:
    """Normaliza um JSON Schema para o modo *strict* de structured outputs.

    Azure OpenAI / OpenAI exigem, em ``response_format`` com
    ``strict: True``, que **todo** objeto declare ``additionalProperties:
    false`` e que ``required`` liste **todas** as propriedades — caso
    contrário a API responde ``400 Invalid schema for response_format``
    (vLLM/guided decoding aceita o mesmo schema sem reclamar).

    O SchemaBuilder do front já preenche ``required`` com todos os campos,
    mas não emite ``additionalProperties: false``; schemas colados à mão
    podem faltar ambos. Esta função recursa pela estrutura do schema
    (``properties``, ``items``, combinadores e ``$defs``) e devolve uma
    cópia normalizada, sem mutar a entrada.
    """
    if not isinstance(schema, dict):
        return schema

    out = dict(schema)

    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: _strictify_schema(v) for k, v in props.items()}
        out["required"] = list(out["properties"].keys())
        out["additionalProperties"] = False
    elif out.get("type") == "object":
        # Objeto sem properties declaradas (free-form) — strict ainda exige a flag.
        out["additionalProperties"] = False

    if "items" in out:
        items = out["items"]
        out["items"] = (
            [_strictify_schema(i) for i in items]
            if isinstance(items, list)
            else _strictify_schema(items)
        )

    for combiner in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(combiner), list):
            out[combiner] = [_strictify_schema(s) for s in out[combiner]]

    for defs_key in ("$defs", "definitions"):
        if isinstance(out.get(defs_key), dict):
            out[defs_key] = {k: _strictify_schema(v) for k, v in out[defs_key].items()}

    return out


def _build_response_format(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "schema": _strictify_schema(schema),
            "strict": True,
        },
    }


def _parse_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content or "{}")
    except json.JSONDecodeError:
        return {"_raw_response": content}


def to_dataframeit_kwargs(config: LlmConfig) -> dict[str, Any]:
    """Mapeia :class:`LlmConfig` para kwargs aceitos por ``dataframeit()``.

    DataFrameIt repassa ``model_kwargs`` para ``langchain.init_chat_model``;
    é lá que ``base_url`` (vLLM/Ollama) e ``azure_endpoint`` viram
    parâmetros do cliente subjacente.

    O parâmetro ``provider`` segue a nomenclatura LangChain:

    - ``"openai"`` (cobre OpenAI direto **e** OpenAI-compat self-host
      via ``model_kwargs={"base_url": ...}``).
    - ``"azure_openai"`` para Azure OpenAI.
    - Qualquer outro string é repassado direto pro
      ``init_chat_model`` (``"google_genai"``, ``"anthropic"``, etc.).
    """
    # ``temperature`` vai sempre, mesmo ``None``: o DataFrameIt fixa
    # ``temperature=0`` por default e só um ``None`` explícito faz o
    # ChatOpenAI omitir o parâmetro (modelos de raciocínio rejeitam 0).
    model_kwargs: dict[str, Any] = {"temperature": config.temperature}
    if config.max_tokens is not None:
        model_kwargs["max_tokens"] = config.max_tokens
    if config.timeout is not None:
        model_kwargs["timeout"] = config.timeout
    if config.reasoning_effort is not None:
        model_kwargs["reasoning_effort"] = config.reasoning_effort
    if config.stream:
        model_kwargs["streaming"] = True
    model_kwargs.update(config.extra)

    if config.provider == "openai_compat":
        # langchain init_chat_model não tem provider "openai_compat" — para
        # endpoints OpenAI-compatible, usar "openai" + base_url.
        if config.base_url:
            model_kwargs["base_url"] = config.base_url
        return {
            "provider": "openai",
            "model": config.model,
            "api_key": config.api_key or "EMPTY",
            "model_kwargs": model_kwargs,
        }

    if config.provider == "azure_openai":
        if not config.base_url:
            raise ValueError("azure_openai requer base_url (azure endpoint)")
        if not config.api_version:
            raise ValueError("azure_openai requer api_version")
        model_kwargs["azure_endpoint"] = config.base_url
        model_kwargs["api_version"] = config.api_version
        return {
            "provider": "azure_openai",
            "model": config.model,
            "api_key": config.api_key,
            "model_kwargs": model_kwargs,
        }

    if config.base_url:
        model_kwargs["base_url"] = config.base_url
    return {
        "provider": config.provider,
        "model": config.model,
        "api_key": config.api_key,
        "model_kwargs": model_kwargs,
    }


def call_llm(
    messages: list[dict[str, str]],
    *,
    config: LlmConfig,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Faz uma chamada chat completion e devolve o dict JSON parseado.

    Em caso de JSON inválido, devolve ``{"_raw_response": "..."}`` em
    vez de levantar — assim o caller (worker, pipeline) consegue
    persistir e marcar o documento como falho sem abortar o batch.
    """
    client = _make_client(config)
    response_format = _build_response_format(schema)

    create_kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "response_format": response_format,
    }
    # OpenAI/Azure deprecaram ``max_tokens`` em favor de
    # ``max_completion_tokens`` (modelos de raciocínio rejeitam o antigo com
    # 400). Servidores OpenAI-compat (vLLM, Ollama) seguem com ``max_tokens``.
    if config.provider in ("openai", "azure_openai"):
        create_kwargs["max_completion_tokens"] = config.max_tokens
    else:
        create_kwargs["max_tokens"] = config.max_tokens
    if config.temperature is not None:
        create_kwargs["temperature"] = config.temperature
    if config.reasoning_effort is not None:
        create_kwargs["reasoning_effort"] = config.reasoning_effort
    create_kwargs.update(config.extra)

    if config.stream:
        create_kwargs["stream"] = True
        stream = client.chat.completions.create(**create_kwargs)
        chunks: list[str] = []
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                chunks.append(piece)
        return _parse_json("".join(chunks))

    resp = client.chat.completions.create(**create_kwargs)
    return _parse_json(resp.choices[0].message.content or "")
