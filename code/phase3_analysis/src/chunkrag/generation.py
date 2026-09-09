from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


@runtime_checkable
class Generator(Protocol):
    def answer(self, question: str, context: str | None = None) -> str:
        ...

    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


def build_qa_prompt(question: str, context: str | None) -> str:
    if context:
        return (
            "Answer the question using only the provided context. "
            "If the answer is not supported by the context, answer with 'unanswerable'.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
    return (
        "Answer the question as briefly as possible.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_chat_prompt(messages: list[dict[str, str]]) -> str:
    lines = [
        "You are a helpful, concise assistant.",
        "Respond naturally to greetings and short chat messages.",
        "If the user's message is unclear, ask a brief clarifying question instead of saying unanswerable.",
        "",
        "Conversation:",
    ]
    for message in messages:
        role = message["role"].strip().capitalize()
        content = message["content"].strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_openai_qa_messages(question: str, context: str | None) -> list[dict[str, str]]:
    if context:
        return [
            {
                "role": "system",
                "content": (
                    "You are an extractive question answering assistant. "
                    "Use only the provided context. "
                    "Copy the shortest answer span supported by the context. "
                    "Do not explain your reasoning. "
                    "If the answer is not fully supported, reply with exactly 'unanswerable'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Answer the following question using only the context.\n\n"
                    f"Question: {question}\n\n"
                    "Context passages:\n"
                    f"{context}\n\n"
                    "Return only the answer text with no explanation."
                ),
            },
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are a concise question answering assistant. "
                "Return only a short answer phrase with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                "Return only the answer text."
            ),
        },
    ]


def build_local_qa_messages(
    question: str,
    context: str | None,
    answer_style: str,
) -> list[dict[str, str]]:
    if answer_style == "extractive":
        return build_openai_qa_messages(question, context)
    if answer_style != "complete":
        raise ValueError(f"Unsupported answer style: {answer_style}")
    if context:
        return [
            {
                "role": "system",
                "content": (
                    "You are a grounded question answering assistant. "
                    "Use only the provided context. Give a concise but complete answer "
                    "containing the information needed to resolve the question. "
                    "Do not explain your reasoning or add citations. If the answer is not "
                    "fully supported, reply with exactly 'unanswerable'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Answer the following question using only the context.\n\n"
                    f"Question: {question}\n\n"
                    "Context passages:\n"
                    f"{context}\n\n"
                    "Return only the final answer."
                ),
            },
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are a concise question answering assistant. "
                "Return a complete answer with no explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nReturn only the final answer.",
        },
    ]


def normalize_qa_response(text: str, *, preserve_multiline: bool = False) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    if preserve_multiline:
        cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    else:
        cleaned = cleaned.splitlines()[0].strip()
    cleaned = re.sub(r"^\s*\[\d+\]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(answer|final answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*the answer is\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("\"'`").strip()
    cleaned = re.sub(r"\s+\[\d+\]\s*$", "", cleaned)
    cleaned = re.sub(r"\s*\([^)]*implies[^)]*\)\s*$", "", cleaned, flags=re.IGNORECASE)
    if not preserve_multiline:
        cleaned = cleaned.rstrip(" .;:")

    lowered = cleaned.lower()
    if preserve_multiline:
        if lowered in {"unanswerable", "not answerable", "not supported"}:
            return "unanswerable"
    elif "unanswerable" in lowered or "not answerable" in lowered or "not supported" in lowered:
        return "unanswerable"
    return cleaned


def compress_answer(question: str, answer: str) -> str:
    cleaned = answer.strip()
    if not cleaned or cleaned == "unanswerable":
        return cleaned

    cleaned = re.sub(
        r"\s*\((?:from|see|passage|source)[^)]*\)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = cleaned.rstrip(" .;:,")
    question_lower = question.lower().strip()

    quantity_match = re.search(
        r"\b(?:over\s+|about\s+|approximately\s+|around\s+|at least\s+)?"
        r"(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty)(?:\s+or\s+(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty))?(?:\s+(?:million|billion|thousand|hundred))?",
        cleaned,
        flags=re.IGNORECASE,
    )
    if question_lower.startswith("how many") or "population" in question_lower:
        if quantity_match:
            quantity = quantity_match.group(0).strip()
            if re.search(r"\bacts?\b", cleaned, flags=re.IGNORECASE):
                quantity = f"{quantity} acts"
            return quantity

    if question_lower.startswith("when") or "what year" in question_lower or "in what year" in question_lower:
        date_match = re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b|\b\d{4}\b",
            cleaned,
        )
        if date_match:
            return date_match.group(0)

    if question_lower.startswith("who"):
        subject_match = re.match(
            r"^(.+?)\s+(?:is|was|are|were|has|have|had|did|does|do|disliked|founded|owned|performed|played|"
            r"caused|expanded|includes|include|contains|contain|premiered|defended|lies|lie|located)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if subject_match:
            return subject_match.group(1).strip().strip(",")

    if question_lower.startswith("where"):
        if "start" in question_lower or "begin" in question_lower:
            from_match = re.search(r"\bfrom\s+([^,.;]+?)(?:\s+to\b|$)", cleaned, flags=re.IGNORECASE)
            if from_match:
                return from_match.group(1).strip()
        for pattern in (
            r"\bin\s+(the\s+[^,.;]+|[^,.;]+)",
            r"\bat\s+(the\s+[^,.;]+|[^,.;]+)",
        ):
            location_match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if location_match:
                return location_match.group(1).strip()

    if "owned by" in cleaned.lower():
        owner_match = re.search(r"\bowned by\s+(.+)$", cleaned, flags=re.IGNORECASE)
        if owner_match:
            return owner_match.group(1).strip().strip(",")

    if "founded by" in cleaned.lower():
        founder_match = re.search(r"\bfounded by\s+(.+)$", cleaned, flags=re.IGNORECASE)
        if founder_match:
            return founder_match.group(1).strip().strip(",")

    if "caused" in cleaned.lower():
        caused_match = re.search(r"\bcaused\s+([^,.;]+)", cleaned, flags=re.IGNORECASE)
        if caused_match:
            return caused_match.group(1).strip()

    if question_lower.startswith(("what", "which")):
        for pattern in (
            r"\b(?:is|are|was|were)\s+(?:now\s+)?(?:an?\s+|the\s+)?([^.;]+)",
            r"\bhave\s+(?:an?\s+|the\s+)?([^.;]+)",
            r"\bmeans\s+([^.;]+)",
            r"\brefers to\s+([^.;]+)",
            r"\brestrain(?:s|ed)?\s+(?:an?\s+|the\s+)?([^.;]+)",
        ):
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

    return cleaned


def build_answer_refinement_messages(
    question: str,
    draft_answer: str,
    context: str | None,
) -> list[dict[str, str]]:
    context_block = f"Context passages:\n{context}\n\n" if context else ""
    return [
        {
            "role": "system",
            "content": (
                "You compress draft answers into the shortest exact answer span. "
                "Use the draft answer and the provided context. "
                "Return only the minimal answer phrase. "
                "If the draft answer is unsupported, reply with exactly 'unanswerable'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"{context_block}"
                f"Draft answer: {draft_answer}\n\n"
                "Return only the final answer phrase."
            ),
        },
    ]


def should_refine_answer(answer: str) -> bool:
    if not answer or answer == "unanswerable":
        return False
    token_count = len(re.findall(r"\w+", answer))
    if token_count <= 4 and not any(marker in answer for marker in ("[", "]", "(", ")", ":")):
        return False
    if re.fullmatch(
        r"(?:over\s+|about\s+|approximately\s+|around\s+|at least\s+)?"
        r"(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
        r"(?:\s+or\s+(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty))?"
        r"(?:\s+(?:million|billion|thousand|hundred|acts?))?",
        answer,
        flags=re.IGNORECASE,
    ):
        return False
    if token_count >= 7:
        return True
    lowered = answer.lower()
    if any(marker in answer for marker in ("[", "]", "(", ")", ":")):
        return True
    return any(
        phrase in lowered
        for phrase in (
            " is ",
            " are ",
            " was ",
            " were ",
            " owned by ",
            " founded by ",
            " performed by ",
            " played by ",
            " from ",
            " to ",
            " implies ",
        )
    )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_torch_dtype(torch_dtype: str | None):
    if torch_dtype is None:
        return None
    if torch_dtype == "auto":
        return "auto"
    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return getattr(torch, torch_dtype)


class QAGenerator:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "auto",
        max_input_tokens: int = 768,
        max_new_tokens: int = 32,
        torch_dtype: str | None = None,
        use_device_map: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.model_name = model_name
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        config = AutoConfig.from_pretrained(model_name, revision=revision)
        self.is_encoder_decoder = bool(config.is_encoder_decoder)
        model_kwargs = {}
        resolved_dtype = resolve_torch_dtype(torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        if revision is not None:
            model_kwargs["revision"] = revision
        if use_device_map:
            model_kwargs["device_map"] = "auto"
        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        if not use_device_map:
            self.model.to(self.device)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.last_trace: dict[str, object] = {}
        self._last_generation_stats: dict[str, object] = {}

    @torch.inference_mode()
    def generate_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        *,
        add_special_tokens: bool = True,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=add_special_tokens,
        ).to(self.device)
        generation_limit = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        generated = self.model.generate(
            **encoded,
            max_new_tokens=generation_limit,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.is_encoder_decoder:
            output_ids = generated[0]
        else:
            prompt_length = encoded["input_ids"].shape[1]
            output_ids = generated[0][prompt_length:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        eos_token_ids = self.tokenizer.eos_token_id
        if eos_token_ids is None:
            eos_ids: set[int] = set()
        elif isinstance(eos_token_ids, int):
            eos_ids = {eos_token_ids}
        else:
            eos_ids = {int(value) for value in eos_token_ids}
        generated_tokens = int(output_ids.shape[0])
        ended_with_eos = bool(generated_tokens and int(output_ids[-1]) in eos_ids)
        self._last_generation_stats = {
            "generated_tokens": generated_tokens,
            "generation_max_new_tokens": generation_limit,
            "generation_length_capped": generated_tokens >= generation_limit and not ended_with_eos,
        }
        return text.strip()

    def _render_chat(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return build_chat_prompt(messages)

    def _chat_token_count(self, rendered_chat: str) -> int:
        return len(self.tokenizer.encode(rendered_chat, add_special_tokens=False))

    def _prepare_causal_qa_prompt(
        self,
        question: str,
        context: str | None,
        answer_style: str,
    ) -> tuple[str, int, int, bool]:
        full_prompt = self._render_chat(build_local_qa_messages(question, context, answer_style))
        full_prompt_tokens = self._chat_token_count(full_prompt)
        if context is None or full_prompt_tokens <= self.max_input_tokens:
            return full_prompt, full_prompt_tokens, full_prompt_tokens, False

        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        lo, hi = 0, len(context_ids)
        best_prompt = self._render_chat(build_local_qa_messages(question, None, answer_style))
        best_prompt_tokens = self._chat_token_count(best_prompt)
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_context = self.tokenizer.decode(
                context_ids[:mid],
                skip_special_tokens=True,
            ).strip()
            candidate_prompt = self._render_chat(
                build_local_qa_messages(question, candidate_context, answer_style)
            )
            candidate_tokens = self._chat_token_count(candidate_prompt)
            if candidate_tokens <= self.max_input_tokens:
                best_prompt = candidate_prompt
                best_prompt_tokens = candidate_tokens
                lo = mid + 1
            else:
                hi = mid - 1
        return best_prompt, full_prompt_tokens, best_prompt_tokens, True

    @torch.inference_mode()
    def answer(self, question: str, context: str | None = None) -> str:
        return self.answer_with_style(question, context, answer_style="extractive")

    @torch.inference_mode()
    def answer_with_style(
        self,
        question: str,
        context: str | None = None,
        *,
        answer_style: str = "extractive",
        max_new_tokens: int | None = None,
    ) -> str:
        if self.is_encoder_decoder:
            if answer_style == "extractive":
                prompt = build_qa_prompt(question, context)
            elif answer_style == "complete":
                prompt = (
                    "Answer the question completely using only the provided context. "
                    "If the answer is not supported, answer with 'unanswerable'.\n\n"
                    f"Question: {question}\n\nContext:\n{context or ''}\n\nAnswer:"
                )
            else:
                raise ValueError(f"Unsupported answer style: {answer_style}")
            full_prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=True))
            raw = self.generate_from_prompt(prompt, max_new_tokens=max_new_tokens)
            used_prompt_tokens = min(full_prompt_tokens, self.max_input_tokens)
            context_truncated = full_prompt_tokens > self.max_input_tokens
        else:
            prompt, full_prompt_tokens, used_prompt_tokens, context_truncated = (
                self._prepare_causal_qa_prompt(question, context, answer_style)
            )
            raw = self.generate_from_prompt(
                prompt,
                max_new_tokens=max_new_tokens,
                add_special_tokens=False,
            )
        normalized = normalize_qa_response(
            raw,
            preserve_multiline=answer_style == "complete",
        )
        final = (
            compress_answer(question, normalized)
            if answer_style == "extractive"
            else normalized
        )
        self.last_trace = {
            "raw_prediction": raw,
            "full_prompt_tokens": full_prompt_tokens,
            "used_prompt_tokens": used_prompt_tokens,
            "context_truncated": context_truncated,
            "refinement_applied": False,
            "answer_style": answer_style,
            **self._last_generation_stats,
        }
        return final

    def chat(self, messages: list[dict[str, str]]) -> str:
        rendered_chat = False
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rendered_chat = True
            except Exception:
                prompt = build_chat_prompt(messages)
        else:
            prompt = build_chat_prompt(messages)
        return self.generate_from_prompt(prompt, add_special_tokens=not rendered_chat)


class ExtractiveFallbackGenerator:
    def __init__(self, max_sentences: int = 3, max_characters: int = 360) -> None:
        self.max_sentences = max_sentences
        self.max_characters = max_characters

    def answer(self, question: str, context: str | None = None) -> str:
        if not context:
            return "I do not have grounded context yet."

        cleaned_lines: list[str] = []
        for line in context.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("matched queries:"):
                continue
            if stripped.lower().startswith("title:"):
                continue
            if re.match(r"^\[\d+\]\s+title:", stripped, flags=re.IGNORECASE):
                continue
            stripped = re.sub(r"^#+\s*", "", stripped)
            stripped = re.sub(r"^[-*]\s*", "", stripped)
            cleaned_lines.append(stripped)
        cleaned_context = ". ".join(cleaned_lines) if cleaned_lines else context
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned_context)
            if sentence.strip()
        ]
        if not sentences:
            return cleaned_context.strip()[: self.max_characters]

        query_terms = {term for term in re.findall(r"\w+", question.lower()) if len(term) > 2}
        scored: list[tuple[int, float, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_terms = set(re.findall(r"\w+", sentence.lower()))
            if len(sentence_terms) < 4:
                continue
            overlap = len(query_terms & sentence_terms)
            score = overlap + (0.2 if any(term in sentence.lower() for term in query_terms) else 0.0)
            scored.append((index, score, sentence))

        if not scored:
            scored = [
                (index, 0.0, sentence)
                for index, sentence in enumerate(sentences)
            ]

        selected = sorted(scored, key=lambda item: (item[1], -item[0]), reverse=True)[: self.max_sentences]
        selected_indices = {index for index, _, _ in selected}
        ordered_sentences = [sentence for index, _, sentence in scored if index in selected_indices]
        answer = " ".join(ordered_sentences).strip()
        if not answer:
            answer = sentences[0]
        return answer[: self.max_characters].strip()

    def chat(self, messages: list[dict[str, str]]) -> str:
        user_messages = [message["content"] for message in messages if message.get("role") == "user" and message.get("content")]
        if not user_messages:
            return "I do not have a user message yet."
        return self.answer(user_messages[-1], context=None)


class OpenAICompatibleGenerator:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        tokenizer_name: str | None = None,
        max_input_tokens: int = 1_024,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the demo extras to use OpenAI-compatible endpoints.") from exc

        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_trace: dict[str, object] = {}

    def _count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                token_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return len(token_ids)
            except Exception:
                pass
        return sum(len(self.tokenizer.encode(message["content"], add_special_tokens=False)) for message in messages)

    def _truncate_context(
        self,
        question: str,
        context: str | None,
        answer_style: str,
    ) -> tuple[list[dict[str, str]], str | None]:
        messages = build_local_qa_messages(question, context, answer_style)
        if context is None or self._count_chat_tokens(messages) <= self.max_input_tokens:
            return messages, context

        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        lo, hi = 0, len(context_ids)
        best_messages = build_local_qa_messages(question, None, answer_style)
        best_context: str | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            truncated_context = self.tokenizer.decode(context_ids[:mid], skip_special_tokens=True).strip()
            candidate_messages = build_local_qa_messages(
                question,
                truncated_context,
                answer_style,
            )
            if self._count_chat_tokens(candidate_messages) <= self.max_input_tokens:
                best_messages = candidate_messages
                best_context = truncated_context
                lo = mid + 1
            else:
                hi = mid - 1
        return best_messages, best_context

    def _refine_answer(self, question: str, draft_answer: str, context: str | None) -> str:
        refinement_messages = build_answer_refinement_messages(question, draft_answer, context)
        if self._count_chat_tokens(refinement_messages) > self.max_input_tokens:
            refinement_messages = build_answer_refinement_messages(question, draft_answer, None)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=refinement_messages,
            temperature=0.0,
            max_tokens=min(24, self.max_new_tokens),
        )
        refined = normalize_qa_response(response.choices[0].message.content or "")
        if not refined or refined == "unanswerable":
            return draft_answer
        return refined

    def answer(self, question: str, context: str | None = None) -> str:
        return self.answer_with_style(question, context, answer_style="extractive")

    def answer_with_style(
        self,
        question: str,
        context: str | None = None,
        *,
        answer_style: str = "extractive",
        max_new_tokens: int | None = None,
    ) -> str:
        full_messages = build_local_qa_messages(question, context, answer_style)
        full_prompt_tokens = self._count_chat_tokens(full_messages)
        truncated_messages, truncated_context = self._truncate_context(
            question,
            context,
            answer_style,
        )
        used_prompt_tokens = self._count_chat_tokens(truncated_messages)
        generation_limit = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=truncated_messages,
            temperature=self.temperature,
            max_tokens=generation_limit,
        )
        choice = response.choices[0]
        message = choice.message
        raw_prediction = message.content or ""
        answer = normalize_qa_response(
            raw_prediction,
            preserve_multiline=answer_style == "complete",
        )
        refinement_applied = answer_style == "extractive" and should_refine_answer(answer)
        if refinement_applied:
            answer = self._refine_answer(question, answer, truncated_context)
        if answer_style == "extractive":
            final = compress_answer(question, normalize_qa_response(answer))
        else:
            final = answer
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        self.last_trace = {
            "raw_prediction": raw_prediction,
            "full_prompt_tokens": full_prompt_tokens,
            "used_prompt_tokens": used_prompt_tokens,
            "context_truncated": context is not None and truncated_context != context,
            "refinement_applied": refinement_applied,
            "answer_style": answer_style,
            "generated_tokens": completion_tokens,
            "generation_max_new_tokens": generation_limit,
            "generation_length_capped": getattr(choice, "finish_reason", None) == "length",
        }
        return final

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        message = response.choices[0].message
        return (message.content or "").strip()
