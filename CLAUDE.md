# CLAUDE.md — `labdados-core`

Núcleo Python compartilhado entre dois repositórios:

- [`escritorio-servicos`](https://github.com/jtrecenti/escritorio-servicos)
  (backend FastAPI + Container Apps + worker)
- [`labdados-sdk`](https://github.com/lab-dados/labdados-sdk)
  (SDK Python externo — `pip install labdados`)

## A regra de ouro

**Só vem pra cá o que precisa ficar byte-equivalente nos dois lados.**

Se uma função é específica de um deles (cliente HTTP do SDK, worker da fila
do backend, FastAPI app, etc.), ela **não** vem. Nem por DRY.

## Subpacotes hoje

| Subpacote | O que faz | Extras |
|---|---|---|
| `contracts` | Pydantic models compartilhados (`FileMetadata`, `ProcessRequest/Response`, `JobStatus`, `ViabilityForm/Results`, `Verdict`). | (nenhum) |
| `viabilidade` | `analyze_form` + `render_report` (Datajud + juscraper + Quarto). | (nenhum) |
| `estruturacao` | Pipeline LLM via DataFrameIt + readers (`txt/md/docx/csv/xlsx`) + prompts canônicos + `LlmConfig` cobrindo OpenAI/Azure/vLLM/Ollama. | `[estruturacao]` |
| `ocr` | `extract` com 2 engines (`pymupdf-tesseract` / `paddleocr`), `formatters.join_pages`/`build_pages_zip`, descoberta automática do binário Tesseract. | `[ocr-cpu]` ou `[ocr-gpu]` |
| `transcricao` | Formatadores TXT/SRT/VTT compartilhados, helpers de timestamp e `Segment` TypedDict. | (nenhum — engine fica nos consumidores) |
| `embeddings` | `build_chunks` (readers + chunking), `embed` (OpenAI-compatível ou sentence-transformers) e `build_result_zip` (parquet + csv + parâmetros). | `[embeddings]` ou `[embeddings-local]` |
| `anonimizacao` | `anonimizar` — detecta PII via `openai/privacy-filter` (HF Transformers, BIOES) e aplica mascaramento (`categoria`/`asteriscos`/`pseudonimo`). 8 categorias: pessoa, email, telefone, endereço, URL, conta, data, segredo. | `[anonimizacao-cpu]` ou `[anonimizacao-gpu]` |

Para adicionar um subpacote novo, siga o padrão de `viabilidade/`:

- pasta com `__init__.py` re-exportando a API pública,
- `pipeline.py` ou `analyze.py` (lógica pura) + helpers privados em `_*.py`,
- `templates/<nome>.qmd` se houver render Quarto,
- testes em `tests/test_<nome>.py`,
- entrada nova em `pyproject.toml` se trouxer dep pesada (extra opcional).

## O que **não** vem pra cá

- Cliente HTTP do SDK → `labdados-sdk/src/labdados/client.py`.
- Routers FastAPI, `services_catalog`, `email`, `storage` → backend.
- Engines de transcrição pesados (WhisperX, pyannote.audio, ffmpeg
  chunking pra OOM-protection) — divergem entre service e SDK
  intencionalmente.
- vLLM client direto, Container Apps GPU specifics — só no backend.

## Versionamento e compatibilidade

Este pacote tem dois consumidores que vivem no GitHub e deployam em
ritmos diferentes (backend deploya em Azure Container Apps; SDK é um
pacote PyPI). Para evitar surpresas:

- **Bumps de patch (0.1.x)**: bug fix, sem mudança de assinatura. Both
  consumers podem atualizar quando quiserem.
- **Bumps de minor (0.x.0)**: nova função/argumento opcional. Backwards
  compatible.
- **Bumps de major (x.0.0)**: mudança no shape do dict de retorno de
  `analyze_form` ou no schema esperado por `render_report`. Coordenar:
  bumpar aqui → bumpar pin nos dois consumidores no mesmo ciclo de
  release.

Os dois consumidores devem **pinar** este pacote em range `>=0.1,<0.2`
até bater 1.0.

## `juscraper` é dependência direta (desde v0.4.0)

Antes da v0.4.0, `juscraper` ficava em `[juscraper]` extra porque
`analyze_form` chamava DataJud via cliente HTTP próprio (`_datajud.py`)
e só usava juscraper pra cjsg/cjpg. Migrado em v0.4.0 — todos os
caminhos passam por juscraper, que vira dep direta:

- DataJud → `juscraper.scraper("datajud").contar_processos(...)`
  (PR [jtrecenti/juscraper#180](https://github.com/jtrecenti/juscraper/pull/180)).
- Jurisprudência/sentenças → `juscraper.scraper(<sigla>).cjsg/cjpg(...)`.

Eliminou ~120 linhas de cliente HTTP duplicado e o `_DATAJUD_KEY`
hardcoded. Mudou só a assinatura interna; `analyze_form` e
`render_report` ficaram byte-equivalentes.

## Por que o template `.qmd` mora aqui

O template Jinja é parte do contrato: as variáveis que ele consome
(`results.tribunais[*].relation`, `results.verdict`, ...) precisam estar
presentes no dict que `analyze_form` devolve. Manter no mesmo pacote
garante que nunca caímos numa situação de "template antigo + analyze
novo" ou vice-versa.

`importlib.resources.files()` lê o template a partir do wheel — funciona
direto sem precisar dar path absoluto.

## Workflow de mudança

Quando precisar mudar algo aqui:

1. **Mude aqui** (`labdados-core`), com testes.
2. Bumpe `_version.py`.
3. Publique tag → CI publica no PyPI (workflow de release ainda a montar
   — por enquanto `pip install git+https://github.com/lab-dados/labdados-core`).
4. Suba o pin nos dois consumidores em PRs simultâneos.

Não dá pra pular o passo 1 — se você "patchear" diretamente em
`viability_runner.py` ou em `labdados.analise_viabilidade.py`, o outro
lado vai divergir. Sempre passe por aqui.
