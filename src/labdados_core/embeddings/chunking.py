"""Corte de textos longos em trechos para embedding."""

from __future__ import annotations

# ~500 tokens em português: cabe no limite do modelo mais restrito do catálogo
# (Cohere embed v3, 512 tokens) e é um tamanho usual para busca semântica.
DEFAULT_MAX_CHARS = 2000
DEFAULT_OVERLAP = 200


def chunk_text(text: str, *, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Divide ``text`` em trechos de até ``max_chars`` caracteres.

    Os cortes caem preferencialmente em quebra de parágrafo, linha, fim de
    frase ou espaço (nessa ordem), procurando na metade final da janela; só
    corta no meio de uma palavra se não houver nenhum desses. Trechos
    consecutivos compartilham ~``overlap`` caracteres, para que uma ideia
    partida ao meio apareça inteira em pelo menos um deles.

    Texto vazio (ou só espaços) devolve lista vazia; texto curto devolve um
    único trecho.
    """
    if max_chars <= 0:
        raise ValueError("max_chars deve ser positivo")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap deve estar entre 0 e max_chars - 1")

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep, max_chars // 2)
                if cut != -1:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
