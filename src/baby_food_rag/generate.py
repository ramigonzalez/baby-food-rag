"""Answer a question from the hybrid index, in Portuguese, with page links.

The model may use only the retrieved chunks. The citation a reader sees is
the printed page when we have one (MS 2019 printed 84 is PDF page 83). The
link is always ``url#page=`` plus the PDF page, which is what a viewer opens.
A fixed disclaimer is added in code, not left to the model.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from baby_food_rag.index import CorpusIndex, search
from baby_food_rag.manifest import find_manifest, load_manifest
from baby_food_rag.types import Chunk, Hit

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_K = 8
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Published OpenRouter rates for this model, USD per million tokens.
# Used only to turn measured token counts into a dollar figure.
PRICE_PER_MILLION = {
    "openai/gpt-4o-mini": (0.15, 0.60),
}

DISCLAIMER = (
    "Isto cita os guias oficiais indexados. Não é orientação médica e não substitui o pediatra."
)
ABSTAIN = "Os guias oficiais indexados não respondem isso."
CALL_TIMEOUT_S = 60
_RETRY_HTTP = {408, 429, 500, 502, 503, 504}

# Final citation, after positional marks are rewritten. Source ids start with a letter.
_SOURCE_CITE = re.compile(r"\[([A-Za-z][A-Za-z0-9-]*) p\.(\d+)\]")
_SOURCE_LINK = re.compile(r"\[([A-Za-z][A-Za-z0-9-]*) p\.(\d+)\](?!\()")
# The model sometimes copies the chunk slot: [3 p.21] instead of [MS-2019 p.84].
_POSITIONAL = re.compile(r"\[(\d+) p\.(\d+)\]")

_PROMPT = """Você responde em português, para pais, sobre alimentação infantil.
Use somente os trechos abaixo. Não use conhecimento de fora deles.
Cada afirmação factual cita o rótulo do trecho, copiado exatamente, no formato [MS-2019 p.104].
O rótulo está na primeira linha de cada trecho. Não cite a posição na lista.
Não escreva [1 p.N], [3 p.N] nem outro número de ordem.
Abstenha-se somente quando nenhum trecho falar do assunto da pergunta.
Nesse caso escreva somente esta frase, sem mais nada:
{abstain}
Se os trechos respondem só parte da pergunta, responda essa parte com citação
e diga o que os guias não cobrem. Não use a frase de abstenção se houver informação útil.
Se dois guias divergirem, diga a divergência e cite os dois.
Não diagnostique, não indique marca e não dê dose de remédio.

Pergunta: {question}

Trechos:
{context}
"""


class MissingAPIKey(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    def estimated_usd(self, model: str) -> float | None:
        rates = PRICE_PER_MILLION.get(model)
        if rates is None:
            return None
        in_rate, out_rate = rates
        return (self.prompt_tokens * in_rate + self.completion_tokens * out_rate) / 1_000_000


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    abstained: bool
    model: str
    latency_s: float
    usage: Usage = field(default_factory=Usage)
    dropped_citations: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        """Model text plus the fixed disclaimer."""
        return f"{self.text.strip()}\n\n{DISCLAIMER}"


class LLM:
    """Chat completion. Tests pass a fake with the same ``complete`` method."""

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    @property
    def usage(self) -> Usage:
        raise NotImplementedError

    @property
    def model(self) -> str:
        raise NotImplementedError


class OpenRouterLLM(LLM):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        timeout_s: float = CALL_TIMEOUT_S,
        opener: object | None = None,
    ) -> None:
        self._model = model
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise MissingAPIKey("OPENROUTER_API_KEY is not set")
        self._key = key
        self._usage = Usage()
        self._timeout_s = timeout_s
        self._opener = opener or urllib.request.urlopen

    @property
    def model(self) -> str:
        return self._model

    @property
    def usage(self) -> Usage:
        return self._usage

    def complete(self, prompt: str) -> str:
        last: Exception | None = None
        for attempt in range(2):
            try:
                return self._complete_once(prompt)
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in _RETRY_HTTP or attempt == 1:
                    raise RuntimeError(f"OpenRouter HTTP {exc.code}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last = exc
                if attempt == 1:
                    raise RuntimeError("OpenRouter timed out") from exc
        raise RuntimeError("OpenRouter timed out") from last

    def _complete_once(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with self._opener(request, timeout=self._timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        self._usage.add(prompt_tokens, completion_tokens)
        return str(choice).strip()


def ask(
    index: CorpusIndex,
    question: str,
    llm: LLM,
    *,
    k: int = DEFAULT_K,
    embedder: object | None = None,
) -> Answer:
    hits = search(index, question, k=k, retriever="hybrid", embedder=embedder)
    prompt = _PROMPT.format(
        abstain=ABSTAIN,
        question=question.strip(),
        context=format_context(hits),
    )
    started = time.perf_counter()
    raw = llm.complete(prompt).strip()
    latency = time.perf_counter() - started
    text, dropped = normalize_citations(raw, hits)
    text = link_citations(text, hits)
    return Answer(
        question=question,
        text=text,
        hits=hits,
        abstained=is_abstention(text),
        model=llm.model,
        latency_s=latency,
        usage=llm.usage,
        dropped_citations=dropped,
    )


def run_question(
    messages: list[dict],
    question: str,
    index: CorpusIndex,
    load_embedder: object,
    make_llm: object,
    *,
    k: int = DEFAULT_K,
) -> list[dict]:
    """Load the embedder, then answer. A load failure raises with history intact."""
    embedder = load_embedder(index.embedder_name)
    llm = make_llm()
    return append_successful_turn(messages, question, index, llm, k=k, embedder=embedder)


def turn_messages(
    index: CorpusIndex,
    question: str,
    llm: LLM,
    *,
    k: int = DEFAULT_K,
    embedder: object | None = None,
) -> tuple[dict, dict]:
    """Build the user and assistant messages. Raises before either exists."""
    answer = ask(index, question, llm, k=k, embedder=embedder)
    assistant = {
        "role": "assistant",
        "content": format_answer_markdown(answer),
        "passages": passages_markdown(answer.hits),
    }
    return {"role": "user", "content": question}, assistant


def append_successful_turn(
    messages: list[dict],
    question: str,
    index: CorpusIndex,
    llm: LLM,
    *,
    k: int = DEFAULT_K,
    embedder: object | None = None,
) -> list[dict]:
    """Add the question only after the answer exists."""
    user, assistant = turn_messages(index, question, llm, k=k, embedder=embedder)
    return [*messages, user, assistant]


def question_to_send(typed: str | None, failed: str | None, retry: bool) -> str | None:
    """A new question wins. Retry resends the failed text unchanged."""
    if typed and typed.strip():
        return typed.strip()
    if retry and failed and failed.strip():
        return failed.strip()
    return None


def is_abstention(text: str) -> bool:
    body = text.replace(DISCLAIMER, "")
    body = " ".join(body.split())
    target = " ".join(ABSTAIN.split())
    return body == target


def format_context(hits: list[Hit]) -> str:
    """Each chunk starts with the exact citation token. No list index."""
    if not hits:
        return "(nenhum trecho recuperado)"
    blocks = []
    for hit in hits:
        chunk = hit.chunk
        label = cite_label(chunk.source_id, chunk.printed_page, chunk.page)
        blocks.append(f"{label}\n{chunk.citation_url}\n{chunk.text}")
    return "\n\n".join(blocks)


def cite_label(source_id: str, printed_page: str | None, pdf_page: int) -> str:
    shown = printed_page if printed_page else str(pdf_page)
    return f"[{source_id} p.{shown}]"


def shown_page(chunk: Chunk) -> str:
    return chunk.printed_page if chunk.printed_page else str(chunk.page)


def page_matches(chunk: Chunk, page: str) -> bool:
    """True only for the page number shown to the model.

    MS 2019 printed page N sits on PDF page N-1, so a PDF-page match would
    bind a correct printed citation to the neighboring sheet.
    """
    return shown_page(chunk) == page


def citation_marks(text: str) -> list[tuple[str, str]]:
    """[source_id p.N] marks in the final answer, including markdown links."""
    return _SOURCE_CITE.findall(text)


def normalize_citations(text: str, hits: list[Hit]) -> tuple[str, list[str]]:
    """Map [3 p.21] to that chunk's label, and drop cites outside the hits.

    The page on a positional mark is ignored. The slot picks the chunk, and
    the label uses that chunk's printed page. A source cite must use that
    same shown page. The PDF page is only the link target.
    """
    dropped: list[str] = []

    def from_slot(match: re.Match[str]) -> str:
        slot = int(match.group(1))
        if slot < 1 or slot > len(hits):
            dropped.append(match.group(0))
            return ""
        chunk = hits[slot - 1].chunk
        return cite_label(chunk.source_id, chunk.printed_page, chunk.page)

    rewritten = _POSITIONAL.sub(from_slot, text)

    def keep_known(match: re.Match[str]) -> str:
        source_id, page = match.group(1), match.group(2)
        for hit in hits:
            chunk = hit.chunk
            if chunk.source_id == source_id and page_matches(chunk, page):
                return cite_label(chunk.source_id, chunk.printed_page, chunk.page)
        dropped.append(match.group(0))
        return ""

    rewritten = _SOURCE_CITE.sub(keep_known, rewritten)
    return _tidy(rewritten), dropped


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def link_citations(text: str, hits: list[Hit]) -> str:
    """Turn a validated [SOURCE p.N] into a markdown link."""
    lookup: dict[tuple[str, str], str] = {}
    for hit in hits:
        shown = shown_page(hit.chunk)
        lookup[(hit.chunk.source_id, shown)] = hit.chunk.citation_url

    def replace(match: re.Match[str]) -> str:
        url = lookup.get((match.group(1), match.group(2)))
        if url is None:
            return match.group(0)
        return f"[{match.group(1)} p.{match.group(2)}]({url})"

    return _SOURCE_LINK.sub(replace, text)


def format_answer_markdown(answer: Answer) -> str:
    return answer.body


def indexed_guides(meta: dict, manifest_path: Path | None = None) -> list[dict]:
    """Guides present in the index, with titles from the manifest when known."""
    by_id = {source.id: source for source in load_manifest(manifest_path or find_manifest())}
    rows = []
    for item in meta.get("sources", []):
        source = by_id.get(item["id"])
        rows.append(
            {
                "id": item["id"],
                "title": source.title if source else item["id"],
                "year": item.get("year"),
                "url": source.url if source else "",
            }
        )
    return rows


def passages_markdown(hits: list[Hit]) -> str:
    if not hits:
        return "_Nenhum trecho recuperado._"
    blocks = []
    for hit in hits:
        chunk = hit.chunk
        label = cite_label(chunk.source_id, chunk.printed_page, chunk.page)
        blocks.append(f"**{label}** — [{chunk.citation_url}]({chunk.citation_url})\n\n{chunk.text}")
    return "\n\n".join(blocks)
