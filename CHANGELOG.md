# Changelog

Todas as mudanças notáveis neste pacote são documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/),
versionamento seguindo [SemVer](https://semver.org/lang/pt-BR/) — ver
"Versionamento e compatibilidade" no `CLAUDE.md`.

## [Unreleased]

## [0.12.0] - 2026-10-01

### Adicionado
- `LlmConfig.reasoning_effort` (opcional) — repassado para `call_llm` e
  `to_dataframeit_kwargs` só quando definido.
- `LlmConfig.temperature` aceita `None` para omitir o parâmetro. Modelos
  de raciocínio (família GPT-5, ex.: `gpt-5.6-luna` no Azure da FGV)
  respondem 400 a qualquer `temperature` diferente da default.

### Alterado
- `call_llm` envia `max_completion_tokens` (em vez de `max_tokens`) para
  os providers `openai` e `azure_openai`; modelos de raciocínio rejeitam
  `max_tokens`. `openai_compat` (vLLM, Ollama) continua com `max_tokens`.
- `to_dataframeit_kwargs` sempre repassa `temperature` (inclusive `None`):
  o DataFrameIt fixa `temperature=0` por default e só um `None` explícito
  faz o ChatOpenAI omitir o parâmetro.

## [0.11.1] - 2026-05-04

### Corrigido
- `anonimizacao._engine.detectar_pii` quebrava com `RuntimeError: tensor
  size mismatch (2040 vs 512)` ao processar textos longos com modelos
  BERT (LeNER-Br). O tokenizer do BERT-base reporta
  `model_max_length = 1e30` (sentinela "infinito" do HF), o que fazia o
  clamp da janela cair no fallback default 2048 — estourando o
  `max_position_embeddings = 512` do modelo. Agora o engine consulta
  primeiro `model.config.max_position_embeddings` (fonte canônica) e
  só depois cai no `tokenizer.model_max_length`.

## [0.11.0] - 2026-05-04

### Adicionado
- Suporte ao modelo `pierreguillou/ner-bert-base-cased-pt-lenerbr` no
  pipeline de anonimização — BERT base PT-BR fine-tuned em LeNER-Br
  (decisões judiciais brasileiras), 110M params, 6 categorias
  (`PESSOA`, `ORGANIZACAO`, `LOCAL`, `TEMPO`, `LEGISLACAO`,
  `JURISPRUDENCIA`). Alternativa rápida e especializada em texto
  jurídico ao `openai/privacy-filter` (multilíngue, 8 categorias PII).
- `LABEL_PT` em `anonimizacao.strategies` ganhou entradas explícitas
  pras categorias do LeNER-Br (`TEMPO` → `DATA`; demais idem).

### Mudado
- Engine `anonimizacao._engine.detectar_pii`:
  - Clampa o tamanho da janela ao `tokenizer.model_max_length`. Permite
    rodar BERT-base (max 512 posições) sem estourar o positional
    embedding; `openai/privacy-filter` (128k) continua usando 2048.
  - Liga `add_special_tokens=True` automaticamente quando o tokenizer
    tem `cls_token_id` (BERT-style). Sem isso, a primeira palavra do
    texto ficava colada no embedding posicional 0 (treinado pra `[CLS]`)
    e o modelo errava a label dela.
  - Pós-processamento `_merge_contiguous` mescla spans adjacentes de
    mesma label separados só por espaço/pontuação intra-token (`/`,
    `.`, `-`, `_`). Resolve fragmentação típica de NER PT-BR — datas
    (`12/05/2023`), siglas (`TJSP`), CNPJs (`XYZ S/A`) — que vinham em
    spans separados por causa da tokenização sub-word.

## [0.10.0] - 2026-05-03

### Adicionado
- Subpacote `labdados_core.anonimizacao` — detecção e mascaramento de
  PII com `openai/privacy-filter` via HF Transformers (CPU/CUDA).
  Pipeline de alto nível em `pipeline.anonimizar(...)` + estratégias
  `categoria` / `asteriscos` / `pseudonimo` em `strategies`.

## [0.9.1] - 2026-05-03

### Mudado
- **Primeira release no PyPI.** `juscraper @ git+https://...` virou
  `juscraper>=0.3,<1.0` em `[project.dependencies]` (juscraper 0.3.0 saiu
  hoje no PyPI, com os filtros do PR #180 incluídos).
- Removido `[tool.hatch.metadata] allow-direct-references = true` —
  não há mais direct refs.
- README e Changelog passam a apontar para `lab-dados/...` (org real)
  em vez de `labdados/...` (não existente).

### Adicionado
- `RELEASING.md` com procedimento de release via Trusted Publisher.
- `.github/workflows/release.yml` — publica em PyPI via OIDC quando
  uma tag `v*` é empurrada.

## [0.9.0] - 2026-05-02

### Adicionado
- Subpacote `labdados_core.transcricao` — formatters compartilhados
  entre o SDK e `services/transcription`:
  - `Segment` (TypedDict) — formato comum entre engines.
  - `format_timestamp_srt` / `format_timestamp_vtt` — incluem ms.
  - `format_segments(segments, *, output_format, include_timestamps,
    with_speaker)` — devolve string única em txt/srt/vtt, com suporte
    opcional a speaker (diarização).

### Notas
- Cobre apenas o que estava **acidentalmente** duplicado entre os dois
  lados (formato + helpers de timestamp). O **engine** de transcrição
  fica onde está: o SDK usa faster-whisper simples; o serviço usa
  faster-whisper com chunking ffmpeg para OOM-protection + WhisperX
  com diarização pyannote. Cada lado tem complexidade própria que
  justifica não unificar agora.
- Sem novos extras opcionais — formatters são puro Python.
- Quem migrava da versão do SDK ganha **milissegundos** no SRT (era
  truncado em segundos antes). Mais detalhado, não quebra parsers.

## [0.8.0] - 2026-05-02

### Adicionado
- Subpacote `labdados_core.ocr` — pipeline de OCR compartilhado entre
  o SDK (modo local) e o serviço `services/ocr` no escritório:
  - `pipeline.extract(pdf, *, modelo, languages, dpi, deskew,
    bw_fallback, use_gpu)` — recebe bytes/Path, devolve `list[str]`
    (uma string por página). Tenta texto nativo primeiro; só roda OCR
    em páginas sem texto embutido.
  - Engines via lazy import por extra:
    - `ocr-cpu`: PyMuPDF + pytesseract (`modelo="pymupdf-tesseract"`).
    - `ocr-gpu`: PaddleOCR 3.x (`modelo="paddleocr"`).
  - `bw_fallback` (default `True`): re-OCR em preto-e-branco quando o
    Tesseract devolve vazio — resgata scans de baixo contraste sem
    pagar opencv.
  - `formatters.join_pages(pages, *, output_format)` e
    `formatters.build_pages_zip(files_data, *, output_format)` —
    consolidam o empacotamento (txt único pro SDK, zip por página pro
    serviço).
  - `_tesseract.configure_tesseract_command()` — discovery do binário
    no Windows (env `TESSERACT_CMD`, `which`, fallback em
    `C:\Program Files\Tesseract-OCR`).
  - Exceções: `EngineUnavailable` (extra ausente),
    `TesseractNotFound` (binário ausente).
- Novos extras `[ocr-cpu]` e `[ocr-gpu]`.

### Notas
- `services/ocr/main.py` e `labdados.ocr` ainda usam suas
  implementações próprias — migração para chamar o core fica para PR
  posterior. O pipeline é byte-equivalente ao que ambos faziam.
- `numpy` voltou a ser sem pin (era `<2` no `[ocr-gpu]`); paddleocr
  3.x aceita numpy 2.x e o pin colidia com pandas (puxado por
  `[estruturacao]`).

## [0.7.0] - 2026-05-02

### Mudado
- `labdados_core.estruturacao.estruturar` passa a usar
  [DataFrameIt](https://brunodcdo.com.br/dataframeit/) internamente.
  Ganha paralelização (`parallel_requests=N`), retry com backoff, rate-limit
  detection e structured output via Pydantic — tudo built-in.
- Schema agora aceita **`BaseModel` Pydantic** ou JSON Schema dict
  (convertido em Pydantic dinamicamente via novo
  `schema_utils.ensure_pydantic_model`). A superfície atual do SDK
  (`estruturacao(schema=dict, ...)`) continua funcionando sem mudança
  para o usuário final.
- Novo helper `_llm.to_dataframeit_kwargs(LlmConfig)` mapeia a config
  unificada para os kwargs que `dataframeit()` espera. Mantém OpenAI,
  Azure OpenAI e endpoints OpenAI-compatible (vLLM/Ollama/LM Studio,
  via `provider="openai_compat"` que vira `provider="openai"` +
  `model_kwargs={"base_url": ...}` no LangChain).

### Adicionado
- `schema_utils.ensure_pydantic_model` + `UnsupportedSchema`. Cobre
  `string/integer/number/boolean`, `array` com items simples, `object`
  aninhado, e `enum` (vira `Literal`). Construções avançadas
  (`anyOf/oneOf/$ref`) levantam erro com mensagem útil.
- Novo extra dep `dataframeit[openai]>=0.6` em `[estruturacao]`.
- Tests de integração em `tests/test_estruturacao_integration.py`
  rodam com `OPENAI_API_KEY` real (skipa se ausente). Marker
  `pytest.mark.integration` permite filtrar com `-m "not integration"`.
- `.env` adicionado ao `.gitignore`.

### Mantido (sem mudança de API)
- `call_llm`, `build_messages`, `read_document`, `LlmConfig` continuam
  exportados e funcionando — `services/structuring/main.py` ainda os
  consome diretamente. Migração pra `estruturar()` (que ganha o
  paralelismo) fica para um PR posterior.

## [0.6.0] - 2026-05-02

### Adicionado
- Subpacote `labdados_core.estruturacao` — núcleo compartilhado entre
  `services/structuring` e `labdados.estruturacao` (modo local). Substitui
  os prompts/readers/cliente OpenAI duplicados nos dois lados:
  - `prompts.build_messages(system, text, schema, *, schema_position)` —
    posição do schema configurável (`"user"` default, `"system"` legacy SDK).
  - `_llm.LlmConfig` + `_llm.call_llm(...)` — cliente unificado para
    OpenAI, Azure OpenAI e endpoints OpenAI-compatible (vLLM, Ollama,
    LM Studio). Suporta streaming (necessário no caminho Container Apps
    → vLLM) e `response_format=json_schema` (structured outputs / guided
    decoding) quando há schema válido.
  - `readers.read_document(content, filename, *, csv_text_column)` — lê
    `.txt/.md/.docx/.csv/.xlsx` para `[(doc_id, texto), ...]`.
  - `pipeline.estruturar(textos, *, schema, system_prompt, llm_config)` —
    orquestra read → prompt → LLM → parse. Erros viram `{"_error": ...}`
    no resultado em vez de levantar.
- Novo extra `labdados-core[estruturacao]` (`openai>=1.40`, `openpyxl>=3.1`).

### Notas
- Backend e SDK ainda redefinem essa lógica localmente; consumir o core
  em ambos é a próxima fase (PRs 2/PR 3 do plano).
- Avaliação de [DataFrameIt](https://brunodcdo.com.br/dataframeit/) como
  núcleo futuro está em pausa — issue de discussão em aberto, ver
  `escritorio-servicos/.dev-notes/dataframeit-issue-draft.md`.

## [0.5.0] - 2026-05-02

### Mudado (breaking)
- `requires-python` apertado de `>=3.10` para `>=3.11`. `juscraper` (dep
  direta desde 0.4.0) já exigia 3.11 — o pin antigo não era resolvível.

### Adicionado
- Subpacote `labdados_core.contracts` com Pydantic v2 models compartilhados
  entre backend (`escritorio-servicos`) e SDK (`labdados-sdk`):
  - `contracts.files`: `FileMetadata`, `FileInfo`.
  - `contracts.jobs`: `ProcessRequest`, `ProcessResponse`, `JobStatusResponse`,
    `JobStatus` (enum).
  - `contracts.viability`: `ViabilityForm`, `ViabilityResults`, `Verdict` (enum).
- `pydantic>=2.7` como dependência direta.
- `pytest-cov` em `[dev]` para acompanhar cobertura.

### Notas de migração
- Backend e SDK podem migrar incrementalmente: importar de
  `labdados_core.contracts` no lugar de redefinir cada model. Os schemas são
  superset-compatible com os atuais (adicionam campos opcionais que já
  existiam em alguns lugares e em outros não).

## [0.4.0] - 2026-04

### Mudado
- `juscraper` agora é dependência direta (não mais extra opcional).
  `analyze_form` chama Datajud via `juscraper.scraper("datajud")` em vez
  do cliente HTTP próprio `_datajud.py`. Eliminou ~120 linhas de
  duplicação e o `_DATAJUD_KEY` hardcoded.
- Requer PR
  [jtrecenti/juscraper#180](https://github.com/jtrecenti/juscraper/pull/180)
  (`data_ajuizamento_inicio/fim` em `contar_processos`). Pinado no `main`
  até sair release com `>=0.3`.

### Sem mudança de comportamento
- `analyze_form` e `render_report` ficaram byte-equivalentes ao
  comportamento anterior.
