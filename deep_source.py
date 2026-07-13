from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable

from langchain_core.documents import Document


ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass
class DeepRetrievalResult:
    answer: str
    documents: list[Document]
    query_variants: list[str]
    sufficient: bool
    confidence: float
    missing_evidence: str
    stages: int
    candidate_count: int
    evidence_labels: list[str] = field(default_factory=list)


def _content_key(document: Document) -> str:
    source = str(document.metadata.get("source", ""))
    page = document.metadata.get("page", "")
    content = re.sub(r"\s+", " ", document.page_content.strip())
    return f"{source}|{page}|{content[:300]}"


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[\w\u0600-\u06FF]+", text.lower())
    stop_words = {
        "في", "من", "إلى", "على", "عن", "ما", "ماذا", "هل", "هو", "هي",
        "هذا", "هذه", "ذلك", "تلك", "مع", "و", "أو", "ثم", "كيف", "لماذا",
        "the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are",
        "what", "how", "why", "does", "do",
    }
    return {word for word in words if len(word) > 1 and word not in stop_words}


def _safe_json_object(text: str) -> dict:
    text = text.strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


def _safe_json_list(text: str) -> list[str]:
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, dict) and isinstance(data.get("queries"), list):
            return [
                str(item).strip()
                for item in data["queries"]
                if str(item).strip()
            ]
    except json.JSONDecodeError:
        pass

    obj = _safe_json_object(text)
    if isinstance(obj.get("queries"), list):
        return [
            str(item).strip()
            for item in obj["queries"]
            if str(item).strip()
        ]

    return []


class DeepSourceEngine:
    """
    استرجاع متعدد المراحل من المصادر فقط.

    المراحل:
    1. توليد صيغ متعددة للسؤال.
    2. بحث دلالي لكل صيغة.
    3. بحث لفظي داخل المقاطع المخزنة.
    4. توسيع الصفحات المجاورة.
    5. ترتيب مرشحين هجين.
    6. فحص كفاية الأدلة.
    7. جولة استرجاع ثانية عند نقص الدليل.
    8. توليد إجابة موثقة بعلامات مصادر.
    """

    def __init__(
        self,
        *,
        llm,
        semantic_k: int = 12,
        final_k: int = 18,
        query_variant_count: int = 4,
        lexical_k: int = 12,
        neighbor_radius: int = 1,
        max_context_chars: int = 80_000,
        evidence_threshold: float = 0.65,
        max_lexical_scan: int = 5000,
    ):
        self.llm = llm
        self.semantic_k = max(2, semantic_k)
        self.final_k = max(4, final_k)
        self.query_variant_count = max(2, query_variant_count)
        self.lexical_k = max(2, lexical_k)
        self.neighbor_radius = max(0, neighbor_radius)
        self.max_context_chars = max(10_000, max_context_chars)
        self.evidence_threshold = min(max(evidence_threshold, 0.0), 1.0)
        self.max_lexical_scan = max(100, max_lexical_scan)

    async def _progress(
        self,
        callback: ProgressCallback | None,
        message: str,
    ) -> None:
        if callback:
            try:
                await callback(message)
            except Exception:
                pass

    async def generate_query_variants(
        self,
        query: str,
        history_text: str = "",
    ) -> list[str]:
        prompt = (
            "أنت وحدة إعادة صياغة لاسترجاع معلومات من مصادر جامعية. "
            f"أنشئ {self.query_variant_count} صيغ بحث مختلفة ومتكاملة للسؤال. "
            "يجب أن تشمل: صيغة مباشرة، صيغة بالمفاهيم الرئيسة، "
            "وصيغة تبحث عن الأدلة أو التعريفات أو المقارنات ذات الصلة. "
            "لا تجب عن السؤال. أعد JSON فقط بهذا الشكل: "
            '{"queries":["...","..."]}.\n\n'
            f"سياق الحوار المختصر:\n{history_text or 'لا يوجد'}\n\n"
            f"السؤال:\n{query}"
        )

        result = await self.llm.ainvoke(prompt)
        variants = _safe_json_list(str(result.content))

        combined = [query]
        for item in variants:
            if item.lower() not in {q.lower() for q in combined}:
                combined.append(item)

        return combined[: self.query_variant_count + 1]

    def _get_store_data(
        self,
        store,
        selected_source: str | None,
    ) -> tuple[list[str], list[dict]]:
        kwargs = {"include": ["documents", "metadatas"]}
        if selected_source:
            kwargs["where"] = {"source": selected_source}

        data = store.get(**kwargs)
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        if len(documents) > self.max_lexical_scan:
            documents = documents[: self.max_lexical_scan]
            metadatas = metadatas[: self.max_lexical_scan]

        return documents, metadatas

    async def _semantic_candidates(
        self,
        store,
        queries: list[str],
        selected_source: str | None,
        *,
        k: int | None = None,
    ) -> dict[str, dict]:
        candidates: dict[str, dict] = {}
        search_k = k or self.semantic_k

        for query_index, query in enumerate(queries):
            kwargs = {}
            if selected_source:
                kwargs["filter"] = {"source": selected_source}

            documents = await asyncio.to_thread(
                store.similarity_search,
                query,
                search_k,
                **kwargs,
            )

            for rank, document in enumerate(documents):
                key = _content_key(document)
                item = candidates.setdefault(
                    key,
                    {
                        "document": document,
                        "score": 0.0,
                        "semantic_hits": 0,
                        "lexical_score": 0.0,
                        "neighbor": False,
                    },
                )
                reciprocal_rank = 1.0 / (rank + 1)
                query_weight = 1.0 / (1.0 + 0.08 * query_index)
                item["score"] += 2.2 * reciprocal_rank * query_weight
                item["semantic_hits"] += 1

        return candidates

    async def _lexical_candidates(
        self,
        store,
        queries: list[str],
        selected_source: str | None,
    ) -> list[tuple[Document, float]]:
        documents, metadatas = await asyncio.to_thread(
            self._get_store_data,
            store,
            selected_source,
        )

        query_token_sets = [_tokens(query) for query in queries]
        results: list[tuple[Document, float]] = []

        for index, text in enumerate(documents):
            if not text:
                continue

            doc_tokens = _tokens(text)
            if not doc_tokens:
                continue

            best_score = 0.0
            for query_tokens in query_token_sets:
                if not query_tokens:
                    continue
                overlap = len(query_tokens & doc_tokens)
                coverage = overlap / max(1, len(query_tokens))
                precision = overlap / max(1, min(len(doc_tokens), 80))
                score = 0.85 * coverage + 0.15 * precision
                best_score = max(best_score, score)

            if best_score > 0:
                metadata = (
                    metadatas[index]
                    if index < len(metadatas) and metadatas[index]
                    else {}
                )
                results.append(
                    (
                        Document(
                            page_content=text,
                            metadata=metadata,
                        ),
                        best_score,
                    )
                )

        results.sort(key=lambda item: item[1], reverse=True)
        return results[: self.lexical_k]

    async def _expand_neighbors(
        self,
        store,
        candidates: dict[str, dict],
        selected_source: str | None,
    ) -> None:
        if self.neighbor_radius <= 0:
            return

        documents, metadatas = await asyncio.to_thread(
            self._get_store_data,
            store,
            selected_source,
        )

        targets: set[tuple[str, int]] = set()
        for item in candidates.values():
            metadata = item["document"].metadata
            source = str(metadata.get("source", ""))
            page = metadata.get("page")
            if source and isinstance(page, int):
                targets.add((source, page))

        if not targets:
            return

        for index, text in enumerate(documents):
            if not text:
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas) and metadatas[index]
                else {}
            )
            source = str(metadata.get("source", ""))
            page = metadata.get("page")

            if not source or not isinstance(page, int):
                continue

            nearest_distance = None
            for target_source, target_page in targets:
                if source != target_source:
                    continue
                distance = abs(page - target_page)
                if distance <= self.neighbor_radius:
                    nearest_distance = (
                        distance
                        if nearest_distance is None
                        else min(nearest_distance, distance)
                    )

            if nearest_distance is None:
                continue

            document = Document(page_content=text, metadata=metadata)
            key = _content_key(document)

            if key not in candidates:
                candidates[key] = {
                    "document": document,
                    "score": 0.22 / (nearest_distance + 1),
                    "semantic_hits": 0,
                    "lexical_score": 0.0,
                    "neighbor": True,
                }

    def _merge_lexical(
        self,
        candidates: dict[str, dict],
        lexical_results: list[tuple[Document, float]],
    ) -> None:
        for document, lexical_score in lexical_results:
            key = _content_key(document)
            item = candidates.setdefault(
                key,
                {
                    "document": document,
                    "score": 0.0,
                    "semantic_hits": 0,
                    "lexical_score": 0.0,
                    "neighbor": False,
                },
            )
            item["lexical_score"] = max(
                item["lexical_score"],
                lexical_score,
            )
            item["score"] += 1.3 * lexical_score

    def _rank_candidates(
        self,
        candidates: dict[str, dict],
        query: str,
    ) -> list[Document]:
        query_tokens = _tokens(query)

        for item in candidates.values():
            document = item["document"]
            metadata = document.metadata
            page = metadata.get("page")

            if item["semantic_hits"] > 1:
                item["score"] += min(0.5, 0.12 * item["semantic_hits"])

            title_text = " ".join(
                str(metadata.get(key, ""))
                for key in ("source", "title", "section", "chapter")
            )
            title_overlap = len(query_tokens & _tokens(title_text))
            item["score"] += 0.08 * title_overlap

            if isinstance(page, int):
                item["score"] += 0.01

        ranked = sorted(
            candidates.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return [
            item["document"]
            for item in ranked[: self.final_k]
        ]

    def _build_evidence_context(
        self,
        documents: list[Document],
    ) -> tuple[str, list[str]]:
        pieces: list[str] = []
        labels: list[str] = []
        total_chars = 0

        for index, document in enumerate(documents, start=1):
            source = str(
                document.metadata.get("source", "مصدر غير معروف")
            )
            page = document.metadata.get("page")
            location = source
            if isinstance(page, int):
                location += f" — صفحة {page + 1}"

            label = f"S{index}"
            block = (
                f"[{label}] المصدر: {location}\n"
                f"{document.page_content.strip()}"
            )

            if total_chars + len(block) > self.max_context_chars:
                break

            pieces.append(block)
            labels.append(f"[{label}] {location}")
            total_chars += len(block)

        return "\n\n---\n\n".join(pieces), labels

    async def assess_evidence(
        self,
        query: str,
        documents: list[Document],
    ) -> dict:
        context, _ = self._build_evidence_context(documents)

        prompt = (
            "قيّم ما إذا كانت الأدلة التالية كافية للإجابة عن السؤال "
            "اعتمادًا على المصادر فقط. لا تجب عن السؤال. "
            "أعد JSON صالحًا فقط بالشكل التالي:\n"
            '{"sufficient":true,"confidence":0.0,'
            '"missing_evidence":"",'
            '"follow_up_queries":["..."]}\n'
            "confidence رقم بين 0 و1. اجعل sufficient=true فقط عندما "
            "تغطي الأدلة جوهر السؤال ولا تتطلب تخمينًا.\n\n"
            f"السؤال:\n{query}\n\n"
            f"الأدلة:\n{context}"
        )

        result = await self.llm.ainvoke(prompt)
        data = _safe_json_object(str(result.content))

        sufficient = bool(data.get("sufficient", False))
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        missing = str(data.get("missing_evidence", "")).strip()
        follow_up = data.get("follow_up_queries", [])
        if not isinstance(follow_up, list):
            follow_up = []

        return {
            "sufficient": sufficient and confidence >= self.evidence_threshold,
            "confidence": min(max(confidence, 0.0), 1.0),
            "missing_evidence": missing,
            "follow_up_queries": [
                str(item).strip()
                for item in follow_up
                if str(item).strip()
            ][:3],
        }

    async def generate_answer(
        self,
        query: str,
        documents: list[Document],
        assessment: dict,
        history_text: str,
    ) -> tuple[str, list[str]]:
        context, labels = self._build_evidence_context(documents)

        sufficiency_note = (
            "الأدلة كافية."
            if assessment["sufficient"]
            else (
                "الأدلة غير مكتملة. قدّم فقط ما تدعمه الأدلة، "
                "واذكر بوضوح ما لا يمكن الجزم به."
            )
        )

        prompt = (
            "أنت مساعد جامعي يجيب حصريًا من الأدلة المرفقة. "
            "لا تستخدم معرفتك العامة ولا تخمّن. "
            "اكتب إجابة عربية واضحة وعميقة ومترابطة. "
            "بعد كل معلومة مهمة ضع علامة المصدر المناسبة مثل [S1]. "
            "يمكنك الاستنتاج فقط إذا كان الاستنتاج مباشرًا من أكثر من دليل، "
            "وعندها سمّه صراحةً «استنتاجًا». "
            "إذا تعارضت المصادر فاذكر التعارض. "
            "إذا كانت الأدلة ناقصة فاذكر حدود الإجابة بدل اختراع معلومات.\n\n"
            f"حالة الأدلة: {sufficiency_note}\n"
            f"الثقة المقدرة: {assessment['confidence']:.2f}\n"
            f"النقص المحتمل: {assessment['missing_evidence'] or 'لا يوجد'}\n\n"
            f"سياق الحوار المختصر:\n{history_text or 'لا يوجد'}\n\n"
            f"السؤال:\n{query}\n\n"
            f"الأدلة:\n{context}"
        )

        result = await self.llm.ainvoke(prompt)
        answer = str(result.content).strip()

        if not answer:
            answer = (
                "لم أتمكن من صياغة إجابة مدعومة من الأدلة المتاحة."
            )

        return answer, labels

    async def retrieve(
        self,
        *,
        store,
        query: str,
        selected_source: str | None = None,
        history_text: str = "",
        progress_callback: ProgressCallback | None = None,
    ) -> DeepRetrievalResult:
        await self._progress(
            progress_callback,
            "🧠 الوضع العميق: إعادة صياغة السؤال للبحث...",
        )

        query_variants = await self.generate_query_variants(
            query,
            history_text,
        )

        await self._progress(
            progress_callback,
            "🔎 الوضع العميق: البحث الدلالي واللفظي متعدد المسارات...",
        )

        candidates = await self._semantic_candidates(
            store,
            query_variants,
            selected_source,
        )
        lexical = await self._lexical_candidates(
            store,
            query_variants,
            selected_source,
        )
        self._merge_lexical(candidates, lexical)

        await self._progress(
            progress_callback,
            "📄 الوضع العميق: توسيع الصفحات والمقاطع المجاورة...",
        )
        await self._expand_neighbors(
            store,
            candidates,
            selected_source,
        )

        ranked_documents = self._rank_candidates(
            candidates,
            query,
        )

        await self._progress(
            progress_callback,
            "🧪 الوضع العميق: فحص كفاية الأدلة...",
        )
        assessment = await self.assess_evidence(
            query,
            ranked_documents,
        )
        stages = 1

        if not assessment["sufficient"]:
            follow_up_queries = assessment["follow_up_queries"]
            if assessment["missing_evidence"]:
                follow_up_queries.append(
                    f"{query} {assessment['missing_evidence']}"
                )

            follow_up_queries = [
                item for item in follow_up_queries if item.strip()
            ][:4]

            if follow_up_queries:
                stages = 2
                await self._progress(
                    progress_callback,
                    "🔁 الأدلة غير كافية: تنفيذ جولة بحث ثانية...",
                )

                second_candidates = await self._semantic_candidates(
                    store,
                    follow_up_queries,
                    selected_source,
                    k=max(self.semantic_k, self.final_k),
                )
                second_lexical = await self._lexical_candidates(
                    store,
                    follow_up_queries,
                    selected_source,
                )
                self._merge_lexical(
                    second_candidates,
                    second_lexical,
                )

                for key, item in second_candidates.items():
                    if key in candidates:
                        candidates[key]["score"] += item["score"]
                        candidates[key]["semantic_hits"] += item[
                            "semantic_hits"
                        ]
                        candidates[key]["lexical_score"] = max(
                            candidates[key]["lexical_score"],
                            item["lexical_score"],
                        )
                    else:
                        candidates[key] = item

                await self._expand_neighbors(
                    store,
                    candidates,
                    selected_source,
                )
                ranked_documents = self._rank_candidates(
                    candidates,
                    query,
                )
                assessment = await self.assess_evidence(
                    query,
                    ranked_documents,
                )
                query_variants.extend(
                    query for query in follow_up_queries
                    if query not in query_variants
                )

        await self._progress(
            progress_callback,
            "✍️ الوضع العميق: صياغة الإجابة الموثقة...",
        )

        answer, evidence_labels = await self.generate_answer(
            query,
            ranked_documents,
            assessment,
            history_text,
        )

        used_documents = ranked_documents[: len(evidence_labels)]

        return DeepRetrievalResult(
            answer=answer,
            documents=used_documents,
            query_variants=query_variants,
            sufficient=assessment["sufficient"],
            confidence=assessment["confidence"],
            missing_evidence=assessment["missing_evidence"],
            stages=stages,
            candidate_count=len(candidates),
            evidence_labels=evidence_labels,
        )
