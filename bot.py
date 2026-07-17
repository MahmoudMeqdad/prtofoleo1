from __future__ import annotations

import asyncio
import ipaddress
import logging
import mimetypes
import os
import re
import shutil
import socket
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from gtts import gTTS
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import IpBlocked, RequestBlocked
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig
from yt_dlp import YoutubeDL
from google import genai
from google.genai import types

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_core import ResearchPlatform
from deep_source import DeepSourceEngine


# =========================================================
# الإعدادات العامة
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError(
        "TELEGRAM_TOKEN غير موجود. أضفه في ملف .env محلياً، "
        "أو في Variables على Railway/Render عند النشر."
    )

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY غير موجود. أضفه في ملف .env محلياً، "
        "أو في Variables على Railway/Render عند النشر."
    )
DOWNLOADS_DIR = BASE_DIR / "downloads"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

research_platform = ResearchPlatform(BASE_DIR / "research.db")

MAX_PDF_SIZE_MB = 20
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "20"))
MAX_WEB_TEXT_CHARS = 100_000
MAX_HISTORY_MESSAGES = 10
TELEGRAM_TEXT_LIMIT = 3900

# النموذج الافتراضي مجاني وبسقف منفصل عن gemini-2.5-flash.
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-lite-latest")
CHAT_MODEL_FALLBACKS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_CHAT_FALLBACKS",
        "gemini-2.0-flash-lite,gemini-2.5-flash,gemini-2.0-flash,gemini-flash-latest",
    ).split(",")
    if model.strip() and model.strip() != CHAT_MODEL
]
EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001",
)
MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))

DEEP_MODE_DEFAULT = os.getenv(
    "DEEP_MODE_DEFAULT",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}

DEEP_SEMANTIC_K = int(os.getenv("DEEP_SEMANTIC_K", "12"))
DEEP_FINAL_K = int(os.getenv("DEEP_FINAL_K", "18"))
DEEP_QUERY_VARIANTS = int(os.getenv("DEEP_QUERY_VARIANTS", "4"))
DEEP_LEXICAL_K = int(os.getenv("DEEP_LEXICAL_K", "12"))
DEEP_NEIGHBOR_RADIUS = int(os.getenv("DEEP_NEIGHBOR_RADIUS", "1"))
DEEP_MAX_CONTEXT_CHARS = int(
    os.getenv("DEEP_MAX_CONTEXT_CHARS", "80000")
)
DEEP_EVIDENCE_THRESHOLD = float(
    os.getenv("DEEP_EVIDENCE_THRESHOLD", "0.65")
)

# إعدادات التلخيص الكامل المرحلي.
FULL_SUMMARY_BATCH_CHARS = max(
    10_000,
    int(os.getenv("FULL_SUMMARY_BATCH_CHARS", "45000")),
)
FULL_SUMMARY_MERGE_CHARS = max(
    10_000,
    int(os.getenv("FULL_SUMMARY_MERGE_CHARS", "50000")),
)
FULL_SUMMARY_MAX_BATCHES = max(
    1,
    int(os.getenv("FULL_SUMMARY_MAX_BATCHES", "100")),
)


# إعدادات التحكم في حصة Gemini Embeddings المجانية.
# الدفعات الصغيرة مع الانتظار تمنع تجاوز 100 طلب في الدقيقة.
EMBEDDING_BATCH_SIZE = max(
    1,
    int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
)
EMBEDDING_BATCH_DELAY = max(
    0.0,
    float(os.getenv("EMBEDDING_BATCH_DELAY", "7")),
)
EMBEDDING_MAX_RETRIES = max(
    1,
    int(os.getenv("EMBEDDING_MAX_RETRIES", "5")),
)
EMBEDDING_RETRY_BASE_DELAY = max(
    1.0,
    float(os.getenv("EMBEDDING_RETRY_BASE_DELAY", "25")),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("notebook_telegram_bot")

# عميل Google GenAI لمعالجة الصوت مباشرة.
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)


def _is_quota_or_overload_error(exc: BaseException) -> bool:
    text = str(exc)
    lowered = text.lower()
    return (
        "RESOURCE_EXHAUSTED" in text
        or "exceeded your current quota" in lowered
        or "UNAVAILABLE" in text
        or "high demand" in lowered
        or ("429" in text and "quota" in lowered)
        or "503" in text
    )


def build_chat_llm(model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=120,
        # تقليل إعادة المحاولة حتى لا تُستهلك الحصة المجانية بسرعة.
        max_retries=1,
    )


class FallbackChatLLM:
    """يحوّل تلقائياً إلى نموذج مجاني بديل عند نفاد الحصة أو ازدحام الخدمة."""

    def __init__(self, models: list[str]) -> None:
        self.models = models
        self._clients = {name: build_chat_llm(name) for name in models}

    async def ainvoke(self, input, config=None, **kwargs):
        last_error: Exception | None = None
        for model_name in self.models:
            try:
                result = await self._clients[model_name].ainvoke(
                    input,
                    config=config,
                    **kwargs,
                )
                if model_name != self.models[0]:
                    logger.info("تم استخدام النموذج الاحتياطي: %s", model_name)
                return result
            except Exception as exc:
                last_error = exc
                if _is_quota_or_overload_error(exc):
                    logger.warning(
                        "تعذر استخدام النموذج %s (%s). تجربة بديل...",
                        model_name,
                        type(exc).__name__,
                    )
                    continue
                raise
        assert last_error is not None
        raise last_error


llm = FallbackChatLLM([CHAT_MODEL, *CHAT_MODEL_FALLBACKS])

deep_source_engine = DeepSourceEngine(
    llm=llm,
    semantic_k=DEEP_SEMANTIC_K,
    final_k=DEEP_FINAL_K,
    query_variant_count=DEEP_QUERY_VARIANTS,
    lexical_k=DEEP_LEXICAL_K,
    neighbor_radius=DEEP_NEIGHBOR_RADIUS,
    max_context_chars=DEEP_MAX_CONTEXT_CHARS,
    evidence_threshold=DEEP_EVIDENCE_THRESHOLD,
)

text_splitter = RecursiveCharacterTextSplitter(
    # مقاطع أكبر تعني عدد طلبات Embeddings أقل.
    chunk_size=3000,
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", "؟ ", "! ", " ", ""],
)

# ذاكرة محادثة مؤقتة؛ تُمسح عند إعادة تشغيل البرنامج
chat_histories: dict[str, list] = {}

# المصدر المحدد حاليًا لكل محادثة.
# None تعني استخدام جميع المصادر.
selected_sources: dict[str, str | None] = {}


# وضع الاسترجاع لكل محادثة: True=Deep Source، False=Precise.
retrieval_modes: dict[str, bool] = {}


def deep_mode_enabled(storage_id: str) -> bool:
    return retrieval_modes.get(storage_id, DEEP_MODE_DEFAULT)


def short_history_text(storage_id: str, limit: int = 6) -> str:
    messages = history_for(storage_id)[-limit:]
    lines: list[str] = []

    for message in messages:
        role = "الطالب" if isinstance(message, HumanMessage) else "المساعد"
        content = str(getattr(message, "content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:1200]}")

    return "\n".join(lines)


# =========================================================
# أدوات مساعدة
# =========================================================

def user_storage_id(update: Update) -> str:
    """معرّف مستقل لكل محادثة، أفضل من الاعتماد على user_id فقط."""
    if not update.effective_chat:
        raise ValueError("لا توجد محادثة صالحة.")
    return str(update.effective_chat.id)


def user_download_dir(storage_id: str) -> Path:
    path = DOWNLOADS_DIR / storage_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_vector_dir(storage_id: str) -> Path:
    path = VECTOR_DB_DIR / storage_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_vector_store(storage_id: str) -> Chroma:
    """يفتح مخزن المستخدم نفسه في كل مرة."""
    return Chroma(
        collection_name=f"notebook_{storage_id.replace('-', 'n')}",
        persist_directory=str(user_vector_dir(storage_id)),
        embedding_function=embeddings,
    )


def database_has_documents(storage_id: str) -> bool:
    try:
        store = get_vector_store(storage_id)
        return store._collection.count() > 0
    except Exception:
        logger.exception("تعذر فحص قاعدة المستخدم %s", storage_id)
        return False


def is_quota_error(exc: Exception) -> bool:
    """يتحقق مما إذا كان الخطأ ناتجًا عن تجاوز حصة Gemini."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "resource_exhausted",
            "quota exceeded",
            "429",
            "rate limit",
            "too many requests",
        )
    )


def extract_retry_seconds(exc: Exception) -> float | None:
    """يحاول استخراج مدة retryDelay من رسالة Google."""
    text = str(exc)

    patterns = [
        r"retryDelay['\"\s:]+(\d+(?:\.\d+)?)s",
        r"retry in\s+(\d+(?:\.\d+)?)s",
        r"please retry in\s+(\d+(?:\.\d+)?)s",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def add_documents(storage_id: str, documents: list[Document]) -> int:
    """
    يقسّم المحتوى ويضيفه إلى Chroma على دفعات صغيرة.

    عند تجاوز حصة Gemini المجانية ينتظر تلقائيًا ثم يعيد المحاولة،
    بدل إيقاف رفع الملف بالكامل من أول خطأ 429.
    """
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        return 0

    store = get_vector_store(storage_id)
    total_added = 0
    total_batches = (
        len(chunks) + EMBEDDING_BATCH_SIZE - 1
    ) // EMBEDDING_BATCH_SIZE

    logger.info(
        "إضافة %s مقطعًا في %s دفعة، حجم الدفعة=%s",
        len(chunks),
        total_batches,
        EMBEDDING_BATCH_SIZE,
    )

    for batch_number, start in enumerate(
        range(0, len(chunks), EMBEDDING_BATCH_SIZE),
        start=1,
    ):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        ids = [str(uuid.uuid4()) for _ in batch]

        for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
            try:
                store.add_documents(
                    documents=batch,
                    ids=ids,
                )
                total_added += len(batch)

                logger.info(
                    "تمت إضافة الدفعة %s/%s (%s مقاطع).",
                    batch_number,
                    total_batches,
                    len(batch),
                )
                break

            except Exception as exc:
                if not is_quota_error(exc):
                    raise

                if attempt >= EMBEDDING_MAX_RETRIES:
                    raise RuntimeError(
                        "تم تجاوز حصة Gemini Embeddings عدة مرات. "
                        "انتظر دقيقة ثم أعد المحاولة، أو فعّل الفوترة "
                        "لرفع حدود الاستخدام."
                    ) from exc

                suggested_delay = extract_retry_seconds(exc)
                wait_seconds = max(
                    EMBEDDING_RETRY_BASE_DELAY * attempt,
                    (suggested_delay or 0) + 2,
                )

                logger.warning(
                    "تجاوز حصة Embeddings في الدفعة %s/%s. "
                    "الانتظار %.1f ثانية، المحاولة %s/%s.",
                    batch_number,
                    total_batches,
                    wait_seconds,
                    attempt,
                    EMBEDDING_MAX_RETRIES,
                )
                time.sleep(wait_seconds)

        # تهدئة ثابتة بين الدفعات لتجنب تجاوز 100 طلب/دقيقة.
        if batch_number < total_batches and EMBEDDING_BATCH_DELAY > 0:
            time.sleep(EMBEDDING_BATCH_DELAY)

    return total_added


async def send_long_text(
    update: Update,
    text: str,
    *,
    parse_mode: str | None = None,
) -> None:
    """يقسم الرسالة الطويلة إلى عدة رسائل دون تجاوز حد تيليجرام."""
    if not update.effective_message:
        return

    text = text.strip()
    if not text:
        return

    while len(text) > TELEGRAM_TEXT_LIMIT:
        cut = text.rfind("\n", 0, TELEGRAM_TEXT_LIMIT)
        if cut < TELEGRAM_TEXT_LIMIT // 2:
            cut = text.rfind(" ", 0, TELEGRAM_TEXT_LIMIT)
        if cut < TELEGRAM_TEXT_LIMIT // 2:
            cut = TELEGRAM_TEXT_LIMIT

        part, text = text[:cut].strip(), text[cut:].strip()
        await update.effective_message.reply_text(
            part,
            parse_mode=parse_mode,
            disable_web_page_preview=True,
        )

    if text:
        await update.effective_message.reply_text(
            text,
            parse_mode=parse_mode,
            disable_web_page_preview=True,
        )


def extract_youtube_id(url: str) -> str | None:
    patterns = [
        r"(?:youtube\.com/watch\?.*?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/(?:shorts|embed)/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }


def ensure_safe_public_url(url: str) -> None:
    """
    يمنع الوصول إلى localhost والشبكات الداخلية.
    مهم عند تشغيل البوت على خادم عام لتقليل مخاطر SSRF.
    """
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("يسمح فقط بروابط HTTP وHTTPS.")

    if not parsed.hostname:
        raise ValueError("الرابط لا يحتوي على اسم نطاق صالح.")

    hostname = parsed.hostname.lower()

    if hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError("روابط localhost غير مسموحة.")

    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise ValueError("تعذر الوصول إلى اسم النطاق.") from exc

    for address in addresses:
        ip_text = address[4][0]
        ip = ipaddress.ip_address(ip_text)

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("روابط الشبكات الداخلية أو الخاصة غير مسموحة.")


def fetch_web_page(url: str) -> tuple[str, str]:
    ensure_safe_public_url(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; NotebookTelegramBot/1.0; "
            "+https://t.me/)"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=(10, 25),
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type:
        raise ValueError("الرابط لا يشير إلى صفحة HTML مدعومة.")

    # فحص الرابط النهائي أيضًا بعد التحويل
    ensure_safe_public_url(response.url)

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(
        ["script", "style", "noscript", "svg", "nav", "footer", "form"]
    ):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else response.url
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    if len(text) < 100:
        raise ValueError("لم أجد نصًا كافيًا داخل الصفحة.")

    return title[:300], text[:MAX_WEB_TEXT_CHARS]


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in text
        or "exceeded your current quota" in text.lower()
        or (
            "429" in text
            and "quota" in text.lower()
        )
    )


def _is_youtube_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, (IpBlocked, RequestBlocked)):
        return True
    text = str(exc)
    return "Too Many Requests" in text or "IpBlocked" in text


def _ytdlp_base_opts() -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"],
            }
        },
    }
    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget

        opts["impersonate"] = ImpersonateTarget("chrome")
    except Exception:
        pass

    cookie_file = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
    if cookie_file and Path(cookie_file).is_file():
        opts["cookiefile"] = cookie_file

    return opts


def _transcript_items_to_text(transcript) -> str:
    parts: list[str] = []
    for item in transcript:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text", "")
        if text:
            parts.append(str(text))
    return " ".join(parts).strip()


def _vtt_to_text(content: str) -> str:
    lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("NOTE")
            or line.startswith("Kind:")
            or line.startswith("Language:")
            or "-->" in line
            or re.fullmatch(r"\d+", line)
        ):
            continue
        cleaned = re.sub(r"<[^>]+>", "", line).strip()
        if cleaned and (not lines or lines[-1] != cleaned):
            lines.append(cleaned)
    return " ".join(lines).strip()


def _build_youtube_transcript_api() -> YouTubeTranscriptApi:
    webshare_user = os.getenv("WEBSHARE_PROXY_USERNAME", "").strip()
    webshare_pass = os.getenv("WEBSHARE_PROXY_PASSWORD", "").strip()
    if webshare_user and webshare_pass:
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=webshare_user,
                proxy_password=webshare_pass,
            )
        )

    http_proxy = (
        os.getenv("YOUTUBE_HTTP_PROXY", "").strip()
        or os.getenv("HTTPS_PROXY", "").strip()
        or os.getenv("HTTP_PROXY", "").strip()
    )
    https_proxy = os.getenv("YOUTUBE_HTTPS_PROXY", "").strip() or http_proxy
    if http_proxy or https_proxy:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=http_proxy or None,
                https_url=https_proxy or None,
            )
        )

    return YouTubeTranscriptApi()


def _fetch_transcript_via_api(video_id: str) -> str:
    preferred_languages = ["ar", "en"]
    api = _build_youtube_transcript_api()
    transcript_list = api.list(video_id)
    available = list(transcript_list)

    if not available:
        raise ValueError("لا يوجد أي تفريغ نصي لهذا الفيديو.")

    last_error: Exception | None = None

    for language_code in preferred_languages:
        try:
            fetched = transcript_list.find_transcript([language_code]).fetch()
            result = _transcript_items_to_text(fetched)
            if result:
                return result
        except Exception as exc:
            last_error = exc

    for transcript in available:
        if not getattr(transcript, "is_translatable", False):
            continue

        translation_codes = {
            getattr(lang, "language_code", None)
            for lang in (getattr(transcript, "translation_languages", None) or [])
        }

        for language_code in preferred_languages:
            if language_code not in translation_codes:
                continue
            try:
                fetched = transcript.translate(language_code).fetch()
                result = _transcript_items_to_text(fetched)
                if result:
                    return result
            except Exception as exc:
                last_error = exc

    for transcript in available:
        try:
            result = _transcript_items_to_text(transcript.fetch())
            if result:
                return result
        except Exception as exc:
            last_error = exc

    if isinstance(last_error, (IpBlocked, RequestBlocked)):
        raise last_error

    if last_error is not None:
        raise last_error

    raise ValueError("تم العثور على تفريغات، لكن تعذر استخراج نص منها.")


def _fetch_transcript_via_ytdlp(video_id: str) -> str:
    preferred_languages = ["ar", "en", "tr"]
    work_dir = DOWNLOADS_DIR / f"yt_subs_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    last_error: Exception | None = None

    try:
        for index, language_code in enumerate(preferred_languages):
            if index > 0:
                time.sleep(2)

            outtmpl = str(work_dir / f"{video_id}.%(ext)s")
            opts = {
                **_ytdlp_base_opts(),
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": [language_code],
                "subtitlesformat": "vtt/best",
                "outtmpl": outtmpl,
                "sleep_interval_subtitles": 1,
            }
            try:
                with YoutubeDL(opts) as ydl:
                    ydl.download([url])
            except Exception as exc:
                last_error = exc
                continue

            subtitle_files = sorted(work_dir.glob(f"{video_id}*.vtt"))
            for subtitle_path in subtitle_files:
                result = _vtt_to_text(
                    subtitle_path.read_text(encoding="utf-8", errors="ignore")
                )
                if result:
                    return result

        if last_error is not None:
            raise last_error
        raise ValueError("تعذر تنزيل ترجمة الفيديو عبر yt-dlp.")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _download_youtube_audio(video_id: str) -> Path:
    work_dir = DOWNLOADS_DIR / f"yt_audio_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(work_dir / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    max_bytes = MAX_AUDIO_SIZE_MB * 1024 * 1024

    opts = {
        **_ytdlp_base_opts(),
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": outtmpl,
        "max_filesize": max_bytes,
    }

    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])

        audio_files = [
            path
            for path in work_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {
                ".m4a",
                ".mp3",
                ".webm",
                ".ogg",
                ".wav",
                ".mp4",
                ".opus",
            }
        ]
        if not audio_files:
            raise ValueError("تعذر تنزيل الصوت من يوتيوب.")

        audio_path = max(audio_files, key=lambda path: path.stat().st_size)
        if audio_path.stat().st_size > max_bytes:
            raise ValueError(
                f"ملف الصوت أكبر من الحد المسموح ({MAX_AUDIO_SIZE_MB} MB)."
            )
        return audio_path
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def fetch_youtube_transcript(video_id: str) -> str:
    """
    يجلب تفريغ فيديو YouTube مع تفضيل العربية والإنجليزية.
    يستخدم youtube-transcript-api أولاً، ثم yt-dlp عند الفشل،
    ثم تفريغ الصوت عبر Gemini كحل أخير.
    """

    errors: list[Exception] = []

    try:
        return _fetch_transcript_via_api(video_id)
    except Exception as exc:
        errors.append(exc)
        logger.warning(
            "فشل جلب التفريغ عبر youtube-transcript-api للفيديو %s: %s",
            video_id,
            exc,
        )

    try:
        result = _fetch_transcript_via_ytdlp(video_id)
        if result:
            return result
    except Exception as exc:
        errors.append(exc)
        logger.warning(
            "فشل جلب التفريغ عبر yt-dlp للفيديو %s: %s",
            video_id,
            exc,
        )

    audio_path: Path | None = None
    try:
        audio_path = _download_youtube_audio(video_id)
        mime_type = detect_audio_mime_type(audio_path, None)
        transcript = transcribe_audio_file(audio_path, mime_type)
        if transcript:
            return transcript
        raise ValueError("تفريغ الصوت لم يُنتج نصاً.")
    except Exception as exc:
        errors.append(exc)
        logger.warning(
            "فشل تفريغ صوت يوتيوب للفيديو %s: %s",
            video_id,
            exc,
        )
    finally:
        if audio_path is not None:
            shutil.rmtree(audio_path.parent, ignore_errors=True)

    messages: list[str] = []
    if any(_is_youtube_rate_limit_error(exc) for exc in errors):
        messages.append(
            "YouTube يحظر طلبات الترجمة من عنوان IP الحالي مؤقتاً "
            "(أو يقيّدها بشدة)."
        )
    if any(_is_quota_error(exc) for exc in errors):
        messages.append(
            "نفدت حصة Gemini المجانية لتفريغ الصوت "
            f"(النموذج: {CHAT_MODEL}). انتظر إعادة التعيين أو استخدم مفتاحاً/خطة أخرى."
        )

    if messages:
        messages.append(
            "اختياري: أضف بروكسي سكني في .env عبر "
            "WEBSHARE_PROXY_USERNAME و WEBSHARE_PROXY_PASSWORD، "
            "أو ملف كوكيز عبر YOUTUBE_COOKIES_FILE."
        )
        raise RuntimeError(" ".join(messages))

    details = " | ".join(str(exc).splitlines()[0] for exc in errors[-3:])
    raise RuntimeError(
        "تعذر جلب تفريغ الفيديو. "
        "تأكد من أن الفيديو عام ويحتوي على ترجمة أو تفريغ نصي. "
        f"التفاصيل: {details}"
    )


def detect_audio_mime_type(
    file_path: Path,
    telegram_mime_type: str | None,
) -> str:
    """يحدد نوع الملف الصوتي لإرساله إلى Gemini."""
    if telegram_mime_type and telegram_mime_type.startswith("audio/"):
        return telegram_mime_type

    guessed, _ = mimetypes.guess_type(file_path.name)
    if guessed and guessed.startswith("audio/"):
        return guessed

    suffix_map = {
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
    }
    return suffix_map.get(file_path.suffix.lower(), "audio/ogg")


def transcribe_audio_file(
    file_path: Path,
    mime_type: str,
) -> str:
    """
    يحول الملف الصوتي إلى نص عربي باستخدام Gemini Audio Understanding.
    يُرسل الملف كبيانات مباشرة، لذلك يقتصر الحجم الافتراضي على 20 MB.
    """
    audio_bytes = file_path.read_bytes()

    prompt = (
        "حوّل هذا التسجيل الصوتي إلى نص مكتوب بدقة. "
        "اكتب الكلام كما قيل دون تلخيص. "
        "حافظ على اللغة الأصلية، وإذا كان الكلام بالعربية فاستخدم العربية. "
        "قسّم النص إلى فقرات واضحة عند تغير المتحدث أو الموضوع. "
        "لا تضف شرحًا أو مقدمة أو تعليقًا خارج التفريغ النصي."
    )

    models_to_try = [CHAT_MODEL, *CHAT_MODEL_FALLBACKS]
    last_error: Exception | None = None

    for model_name in models_to_try:
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=mime_type,
                    ),
                ],
            )
            transcript = (response.text or "").strip()
            if transcript:
                if model_name != CHAT_MODEL:
                    logger.info(
                        "تم تفريغ الصوت باستخدام النموذج الاحتياطي: %s",
                        model_name,
                    )
                return transcript
            last_error = ValueError(
                "لم يتمكن Gemini من استخراج نص من التسجيل الصوتي."
            )
        except Exception as exc:
            last_error = exc
            if _is_quota_error(exc):
                logger.warning(
                    "نفدت حصة النموذج %s أثناء تفريغ الصوت، سيتم تجربة بديل.",
                    model_name,
                )
                continue
            raise

    if last_error is not None:
        raise last_error

    raise ValueError("لم يتمكن Gemini من استخراج نص من التسجيل الصوتي.")



def load_pdf(file_path: Path, original_name: str) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update(
            {
                "source": original_name,
                "source_type": "pdf",
            }
        )

    return docs


def history_for(storage_id: str) -> list:
    return chat_histories.setdefault(storage_id, [])


def trim_history(storage_id: str) -> None:
    history = history_for(storage_id)
    if len(history) > MAX_HISTORY_MESSAGES:
        chat_histories[storage_id] = history[-MAX_HISTORY_MESSAGES:]


def build_sources_text(docs: list[Document]) -> str:
    seen: set[str] = set()
    sources: list[str] = []

    for doc in docs:
        source = str(doc.metadata.get("source", "مصدر غير معروف"))
        page = doc.metadata.get("page")

        label = source
        if isinstance(page, int):
            label += f" — صفحة {page + 1}"

        if label not in seen:
            seen.add(label)
            sources.append(label)

    return "\n".join(f"- {source}" for source in sources[:10])


def get_source_stats(storage_id: str) -> list[dict]:
    """يعيد قائمة المصادر وعدد المقاطع ونوع كل مصدر."""
    store = get_vector_store(storage_id)
    data = store.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    stats: dict[str, dict] = {}

    for metadata in metadatas:
        metadata = metadata or {}
        source = str(metadata.get("source", "مصدر غير معروف"))
        source_type = str(metadata.get("source_type", "unknown"))

        if source not in stats:
            stats[source] = {
                "source": source,
                "source_type": source_type,
                "chunks": 0,
            }

        stats[source]["chunks"] += 1

    return sorted(
        stats.values(),
        key=lambda item: item["source"].lower(),
    )


def resolve_source_reference(
    storage_id: str,
    reference: str,
) -> str | None:
    """
    يحل رقم المصدر أو اسمه إلى الاسم الكامل المخزن.
    يدعم الرقم من أمر /files أو جزءًا فريدًا من الاسم.
    """
    sources = get_source_stats(storage_id)
    reference = reference.strip()

    if not reference:
        return None

    if reference.isdigit():
        index = int(reference) - 1
        if 0 <= index < len(sources):
            return str(sources[index]["source"])
        return None

    exact = [
        item["source"]
        for item in sources
        if item["source"].lower() == reference.lower()
    ]
    if exact:
        return str(exact[0])

    partial = [
        item["source"]
        for item in sources
        if reference.lower() in item["source"].lower()
    ]

    if len(partial) == 1:
        return str(partial[0])

    return None


def selected_source_for(storage_id: str) -> str | None:
    return selected_sources.get(storage_id)


def delete_source(storage_id: str, source: str) -> int:
    """يحذف جميع المقاطع التابعة لمصدر واحد من Chroma."""
    store = get_vector_store(storage_id)
    result = store.get(
        where={"source": source},
        include=[],
    )
    ids = result.get("ids") or []

    if ids:
        store.delete(ids=ids)

    return len(ids)


def source_filter_for(storage_id: str) -> dict | None:
    source = selected_source_for(storage_id)
    return {"source": source} if source else None


def split_text_groups(items: list[str], max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for item in items:
        item = item.strip()
        if not item:
            continue
        if current and current_size + len(item) > max_chars:
            groups.append(current)
            current = []
            current_size = 0
        if len(item) > max_chars:
            for start in range(0, len(item), max_chars):
                if current:
                    groups.append(current)
                    current = []
                    current_size = 0
                groups.append([item[start:start + max_chars]])
            continue
        current.append(item)
        current_size += len(item)
    if current:
        groups.append(current)
    return groups


def get_all_stored_documents(
    storage_id: str,
    source: str | None = None,
) -> list[Document]:
    store = get_vector_store(storage_id)

    get_kwargs = {
        "include": ["documents", "metadatas"],
    }
    if source:
        get_kwargs["where"] = {"source": source}

    data = store.get(**get_kwargs)
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    docs: list[Document] = []
    for index, text in enumerate(documents):
        if not text:
            continue
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        docs.append(Document(page_content=text, metadata=metadata))
    docs.sort(key=lambda d: (
        str(d.metadata.get("source", "")),
        int(d.metadata.get("page", 10**9)) if isinstance(d.metadata.get("page"), int) else 10**9,
    ))
    return docs


def format_document_for_summary(doc: Document, index: int) -> str:
    source = str(doc.metadata.get("source", "مصدر غير معروف"))
    page = doc.metadata.get("page")
    header = f"[المقطع {index} | المصدر: {source}"
    if isinstance(page, int):
        header += f" | الصفحة: {page + 1}"
    header += "]"
    return f"{header}\n{doc.page_content.strip()}"


async def summarize_text_batch(batch_text: str, batch_number: int, total_batches: int) -> str:
    prompt = (
        "أنت محلل مستندات دقيق. لخّص الدفعة التالية من محتوى المصادر "
        "تلخيصًا عربيًا منظمًا ومكثفًا، مع الحفاظ على الحقائق والأفكار "
        "والتعريفات والأرقام والنتائج المهمة. لا تضف أي معلومات من خارج النص. "
        "احذف التكرار واذكر التعارضات الظاهرة. "
        f"هذه الدفعة رقم {batch_number} من أصل {total_batches}.\n\n{batch_text}"
    )
    result = await llm.ainvoke(prompt)
    summary = str(result.content).strip()
    if not summary:
        raise ValueError(f"لم يتم إنشاء ملخص للدفعة {batch_number}.")
    return summary


async def merge_summary_group(summaries: list[str], level: int, group_number: int, total_groups: int) -> str:
    joined = "\n\n---\n\n".join(
        f"[ملخص فرعي {i}]\n{text}" for i, text in enumerate(summaries, start=1)
    )
    prompt = (
        "ادمج الملخصات الفرعية التالية في ملخص عربي واحد متماسك ودقيق. "
        "احذف التكرار، وحافظ على النقاط الجوهرية والأرقام والنتائج، "
        "ولا تضف معلومات غير موجودة. "
        f"مستوى الدمج {level}، المجموعة {group_number} من {total_groups}.\n\n{joined}"
    )
    result = await llm.ainvoke(prompt)
    merged = str(result.content).strip()
    if not merged:
        raise ValueError("فشل دمج مجموعة من الملخصات.")
    return merged


async def create_full_staged_summary(storage_id: str, status_message=None) -> tuple[str, list[Document]]:
    selected_source = selected_source_for(storage_id)
    docs = await asyncio.to_thread(
        get_all_stored_documents,
        storage_id,
        selected_source,
    )
    if not docs:
        raise ValueError("لا توجد مقاطع مخزنة لتلخيصها.")

    formatted = [format_document_for_summary(doc, i) for i, doc in enumerate(docs, start=1)]
    batches = split_text_groups(formatted, FULL_SUMMARY_BATCH_CHARS)
    if len(batches) > FULL_SUMMARY_MAX_BATCHES:
        raise ValueError("عدد دفعات التلخيص كبير جدًا. ارفع FULL_SUMMARY_BATCH_CHARS أو لخص مصادر أقل.")

    partials: list[str] = []
    total_batches = len(batches)
    for batch_number, items in enumerate(batches, start=1):
        if status_message:
            try:
                await status_message.edit_text(
                    "📚 جاري التلخيص الكامل على مراحل...\n"
                    f"تلخيص الدفعة {batch_number} من {total_batches}"
                )
            except Exception:
                pass
        partials.append(await summarize_text_batch("\n\n".join(items), batch_number, total_batches))

    current = partials
    level = 1
    while len(current) > 1:
        groups = split_text_groups(current, FULL_SUMMARY_MERGE_CHARS)
        next_level: list[str] = []
        for group_number, group in enumerate(groups, start=1):
            if status_message:
                try:
                    await status_message.edit_text(
                        "🧩 جاري دمج الملخصات المرحلية...\n"
                        f"المستوى {level} — المجموعة {group_number} من {len(groups)}"
                    )
                except Exception:
                    pass
            next_level.append(await merge_summary_group(group, level, group_number, len(groups)))
        if len(next_level) >= len(current):
            joined = "\n\n---\n\n".join(next_level)
            result = await llm.ainvoke(
                "حوّل النص التالي إلى ملخص نهائي عربي شامل ومنظم، مع حذف التكرار "
                "والمحافظة على الأفكار الأساسية والأرقام والنتائج، دون إضافة معلومات خارجية.\n\n" + joined
            )
            return str(result.content).strip(), docs
        current = next_level
        level += 1

    final_prompt = (
        "أعد صياغة الملخص التالي كملخص نهائي شامل للمصادر كلها. "
        "استخدم عناوين واضحة، وابدأ بنظرة عامة، ثم الأفكار الرئيسة، "
        "ثم أهم التفاصيل والنتائج، ثم خلاصة ختامية. لا تضف معلومات خارج النص.\n\n"
        + current[0]
    )
    result = await llm.ainvoke(final_prompt)
    polished = str(result.content).strip()
    return polished or current[0], docs


def is_full_summary_request(user_query: str) -> bool:
    normalized = re.sub(r"\s+", " ", user_query.strip().lower())
    phrases = (
        "لخص الملف كامل", "لخّص الملف كامل", "تلخيص الملف كامل", "تلخيص كامل",
        "لخص جميع الصفحات", "لخّص جميع الصفحات", "لخص كل الصفحات",
        "لخّص كل الصفحات", "حلل الملف كامل", "حلّل الملف كامل",
        "اقرأ الملف كامل", "جميع الصفحات", "كل الصفحات", "ملخص شامل",
        "summary of the whole file", "summarize the whole file", "summarize all pages",
    )
    return any(phrase in normalized for phrase in phrases)


# =========================================================
# أوامر البوت
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = (
        "📘 <b>مرحبًا بك في مساعد المصادر على تيليجرام</b>\n\n"
        "أرسل لي:\n"
        "• ملف PDF لإضافته إلى مصادرك.\n"
        "• رابط مقال ويب عام.\n"
        "• رابط فيديو YouTube يحتوي على ترجمة أو تفريغ (أي لغة).\n"
        "• رسالة صوتية أو ملف صوت لتحويله إلى نص.\n"
        "• سؤالًا عن المصادر المضافة.\n\n"
        "الأوامر:\n"
        "/sources — عرض عدد المقاطع المخزنة\n"
        "/deep — تفعيل البحث العميق متعدد المراحل\n"
        "/precise — تفعيل البحث السريع المركز\n"
        "/mode — عرض وضع الاسترجاع الحالي\n"
        "/files — عرض قائمة المصادر\n"
        "/select رقم — اختيار مصدر محدد\n"
        "/all — العودة إلى جميع المصادر\n"
        "/delete رقم — حذف مصدر واحد\n"
        "/courses — عرض المساقات المنظمة\n"
        "/seed_demo — إضافة بيانات جامعية تجريبية\n"
        "/feedback 1-5 — تقييم آخر إجابة\n"
        "/research_stats — مؤشرات التجربة\n"
        "/export_logs — تصدير سجل التجارب CSV\n"
        "/summary — تلخيص المصدر المحدد أو جميع المصادر\n"
        "/podcast — إنشاء ملخص صوتي\n"
        "/reset — حذف مصادر هذه المحادثة وذاكرتها\n"
        "/help — عرض التعليمات"
    )
    await update.effective_message.reply_text(
        message,
        parse_mode=ParseMode.HTML,
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await start(update, context)



async def full_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage_id = user_storage_id(update)
    if not database_has_documents(storage_id):
        await update.effective_message.reply_text(
            "⚠️ لا توجد مصادر بعد. أرسل ملف PDF أو رابطًا أو صوتًا أولًا."
        )
        return
    status = await update.effective_message.reply_text("📚 بدأ التلخيص الكامل على مراحل...")
    try:
        summary, docs = await create_full_staged_summary(storage_id, status)
        await status.delete()
        unique_sources = sorted({str(doc.metadata.get("source", "مصدر غير معروف")) for doc in docs})
        sources_text = "\n".join(f"- {source}" for source in unique_sources[:20])
        final_text = "📘 الملخص الكامل للمصادر:\n\n" + summary
        if sources_text:
            final_text += "\n\n📚 المصادر التي شملها التلخيص:\n" + sources_text
        await send_long_text(update, final_text)
    except Exception as exc:
        logger.exception("فشل التلخيص الكامل")
        error_text = str(exc).strip()
        if len(error_text) > 1500:
            error_text = error_text[:1500] + "..."
        await status.edit_text("❌ تعذر إنشاء التلخيص الكامل:\n" + error_text)




async def files_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """يعرض جميع المصادر المخزنة مع أرقامها."""
    storage_id = user_storage_id(update)

    try:
        sources = await asyncio.to_thread(
            get_source_stats,
            storage_id,
        )

        if not sources:
            await update.effective_message.reply_text(
                "⚠️ لا توجد مصادر مخزنة."
            )
            return

        selected = selected_source_for(storage_id)
        lines = ["📚 المصادر المخزنة:\n"]

        type_icons = {
            "pdf": "📄",
            "web": "🌐",
            "youtube": "🎬",
            "audio": "🎧",
        }

        for index, item in enumerate(sources, start=1):
            source = str(item["source"])
            source_type = str(item["source_type"])
            chunks = int(item["chunks"])
            icon = type_icons.get(source_type, "📌")
            marker = " ✅" if source == selected else ""

            lines.append(
                f"{index}. {icon} {source}{marker}\n"
                f"   المقاطع: {chunks}"
            )

        lines.append(
            "\nاستخدم /select رقم لاختيار مصدر، "
            "أو /all للبحث في الجميع."
        )

        await send_long_text(
            update,
            "\n".join(lines),
        )

    except Exception as exc:
        logger.exception("تعذر عرض المصادر")
        await update.effective_message.reply_text(
            f"❌ تعذر عرض المصادر: {exc}"
        )


async def select_source_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """يحدد مصدرًا واحدًا للبحث والتلخيص والبودكاست."""
    storage_id = user_storage_id(update)
    reference = " ".join(context.args).strip()

    if not reference:
        await update.effective_message.reply_text(
            "استخدم الأمر بهذا الشكل:\n"
            "/select 1\n"
            "ثم استخدم /files لمعرفة أرقام المصادر."
        )
        return

    source = await asyncio.to_thread(
        resolve_source_reference,
        storage_id,
        reference,
    )

    if not source:
        await update.effective_message.reply_text(
            "❌ لم أجد مصدرًا مطابقًا. "
            "استخدم /files ثم اختر الرقم الصحيح."
        )
        return

    selected_sources[storage_id] = source
    chat_histories.pop(storage_id, None)

    await update.effective_message.reply_text(
        "✅ تم اختيار المصدر:\n"
        f"{source}\n\n"
        "ستقتصر الأسئلة والتلخيص والبودكاست عليه."
    )


async def all_sources_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """يعيد البحث إلى جميع المصادر."""
    storage_id = user_storage_id(update)
    selected_sources[storage_id] = None
    chat_histories.pop(storage_id, None)

    await update.effective_message.reply_text(
        "✅ تم إلغاء تحديد المصدر. "
        "سيتم البحث الآن في جميع المصادر."
    )


async def delete_source_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """يحذف مصدرًا واحدًا مع جميع مقاطعه."""
    storage_id = user_storage_id(update)
    reference = " ".join(context.args).strip()

    if not reference:
        await update.effective_message.reply_text(
            "استخدم الأمر بهذا الشكل:\n"
            "/delete 1\n\n"
            "استخدم /files لمعرفة رقم المصدر."
        )
        return

    source = await asyncio.to_thread(
        resolve_source_reference,
        storage_id,
        reference,
    )

    if not source:
        await update.effective_message.reply_text(
            "❌ لم أجد مصدرًا مطابقًا."
        )
        return

    deleted_count = await asyncio.to_thread(
        delete_source,
        storage_id,
        source,
    )

    if selected_source_for(storage_id) == source:
        selected_sources[storage_id] = None

    chat_histories.pop(storage_id, None)

    await update.effective_message.reply_text(
        "🗑️ تم حذف المصدر:\n"
        f"{source}\n"
        f"عدد المقاطع المحذوفة: {deleted_count}"
    )




async def deep_mode_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    storage_id = user_storage_id(update)
    retrieval_modes[storage_id] = True
    await update.effective_message.reply_text(
        "✅ تم تفعيل Deep Source Mode.\n"
        "سيستخدم البوت استعلامات متعددة، بحثًا دلاليًا ولفظيًا، "
        "توسيع الصفحات المجاورة، وفحص كفاية الأدلة."
    )


async def precise_mode_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    storage_id = user_storage_id(update)
    retrieval_modes[storage_id] = False
    await update.effective_message.reply_text(
        "✅ تم تفعيل Precise Mode.\n"
        "سيستخدم البوت البحث السريع في المقاطع الأكثر ارتباطًا."
    )


async def retrieval_mode_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    storage_id = user_storage_id(update)
    mode = (
        "Deep Source Mode"
        if deep_mode_enabled(storage_id)
        else "Precise Mode"
    )
    source = selected_source_for(storage_id) or "جميع المصادر"

    await update.effective_message.reply_text(
        f"⚙️ وضع الاسترجاع الحالي: {mode}\n"
        f"📚 نطاق البحث: {source}"
    )


async def courses_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    courses = await asyncio.to_thread(research_platform.list_courses)
    if not courses:
        await update.effective_message.reply_text(
            "لا توجد مساقات منظمة بعد. استخدم /seed_demo أو أضف بيانات الجامعة."
        )
        return

    lines = ["🎓 المساقات المنظمة:"]
    for row in courses:
        instructor = f" — {row['instructor_name']}" if row["instructor_name"] else ""
        lines.append(f"• {row['code']}: {row['name_ar']}{instructor}")
    await send_long_text(update, "\n".join(lines))


async def seed_demo_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    count = await asyncio.to_thread(research_platform.seed_demo_data)
    if count:
        await update.effective_message.reply_text(
            f"✅ تمت إضافة {count} مساقات تجريبية إلى قاعدة البحث."
        )
    else:
        await update.effective_message.reply_text(
            "ℹ️ قاعدة المساقات تحتوي بيانات بالفعل؛ لم تتم إضافة بيانات تجريبية."
        )


async def feedback_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not context.args or not context.args[0].isdigit():
        await update.effective_message.reply_text(
            "استخدم: /feedback 5 تعليق اختياري"
        )
        return

    rating = int(context.args[0])
    if rating < 1 or rating > 5:
        await update.effective_message.reply_text("التقييم يجب أن يكون من 1 إلى 5.")
        return

    comment = " ".join(context.args[1:]).strip()
    session_id = user_storage_id(update)
    await asyncio.to_thread(
        research_platform.add_feedback,
        session_id,
        rating,
        comment,
    )
    await update.effective_message.reply_text("✅ تم تسجيل تقييمك لأغراض البحث.")


async def research_stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    stats = await asyncio.to_thread(research_platform.research_stats)
    routes = "، ".join(
        f"{key}: {value}" for key, value in stats["routes"].items()
    ) or "لا توجد بيانات"
    await update.effective_message.reply_text(
        "📊 مؤشرات المنصة البحثية:\n"
        f"إجمالي الأسئلة: {stats['total_queries']}\n"
        f"متوسط زمن الاستجابة: {stats['average_response_ms']} ms\n"
        f"مرات طلب التوضيح: {stats['clarification_count']}\n"
        f"متوسط تقييم المستخدم: {stats['average_rating']}/5\n"
        f"استعلامات الوضع العميق: {stats['deep_query_count']}\n"
        f"متوسط ثقة الأدلة العميقة: "
        f"{stats['deep_average_confidence']}\n"
        f"نسبة كفاية الأدلة: {stats['deep_sufficiency_rate']}%\n"
        f"مسارات الإجابة: {routes}"
    )


async def export_logs_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    export_path = BASE_DIR / "research_logs.csv"
    count = await asyncio.to_thread(
        research_platform.export_logs_csv,
        export_path,
    )
    with export_path.open("rb") as file:
        await update.effective_message.reply_document(
            document=file,
            filename="research_logs.csv",
            caption=f"📥 سجل التجارب — {count} سجلًا",
        )


async def sources_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    storage_id = user_storage_id(update)

    try:
        store = get_vector_store(storage_id)
        count = store._collection.count()
        selected = selected_source_for(storage_id)
        source_text = (
            f"\n🎯 المصدر المحدد: {selected}"
            if selected
            else "\n🌐 نطاق البحث: جميع المصادر"
        )
        retrieval_text = (
            "\n🧠 الاسترجاع: Deep Source Mode"
            if deep_mode_enabled(storage_id)
            else "\n⚡ الاسترجاع: Precise Mode"
        )
        await update.effective_message.reply_text(
            f"📚 عدد المقاطع النصية المخزنة حاليًا: {count}"
            f"{source_text}"
            f"{retrieval_text}"
        )
    except Exception:
        logger.exception("تعذر قراءة عدد المصادر")
        await update.effective_message.reply_text(
            "❌ تعذر قراءة معلومات المصادر."
        )


async def reset_notebook(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    storage_id = user_storage_id(update)
    vector_path = VECTOR_DB_DIR / storage_id
    download_path = DOWNLOADS_DIR / storage_id

    for path in (vector_path, download_path):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    chat_histories.pop(storage_id, None)
    selected_sources.pop(storage_id, None)
    retrieval_modes.pop(storage_id, None)

    await update.effective_message.reply_text(
        "🧹 تم حذف مصادر هذه المحادثة وذاكرتها بنجاح."
    )


# =========================================================
# استقبال المصادر
# =========================================================

async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    document = message.document if message else None

    if not document:
        return

    file_name = Path(document.file_name or "document.pdf").name

    if document.mime_type != "application/pdf" and not file_name.lower().endswith(
        ".pdf"
    ):
        await message.reply_text("❌ المدعوم حاليًا هو ملفات PDF فقط.")
        return

    size_mb = (document.file_size or 0) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        await message.reply_text(
            f"❌ حجم الملف أكبر من الحد المسموح ({MAX_PDF_SIZE_MB} MB)."
        )
        return

    storage_id = user_storage_id(update)
    file_path = user_download_dir(storage_id) / (
        f"{uuid.uuid4().hex}_{file_name}"
    )

    status = await message.reply_text(
        f"⏳ جاري تنزيل وتحليل الملف: {file_name}\n"
        "قد يستغرق الملف الكبير وقتًا إضافيًا لتجنب تجاوز حصة Google المجانية."
    )

    try:
        telegram_file = await document.get_file()
        await telegram_file.download_to_drive(custom_path=file_path)

        docs = await asyncio.to_thread(load_pdf, file_path, file_name)

        if not docs:
            raise ValueError(
                "لم أتمكن من استخراج نص من الملف؛ قد يكون PDF مصورًا."
            )

        chunk_count = await asyncio.to_thread(
            add_documents,
            storage_id,
            docs,
        )

        await status.edit_text(
            f"✅ تمت إضافة {file_name} بنجاح.\n"
            f"عدد المقاطع النصية الجديدة: {chunk_count}"
        )
    except Exception as exc:
        logger.exception("فشل تحليل PDF")
        error_text = str(exc).strip()
        if len(error_text) > 1500:
            error_text = error_text[:1500] + "..."

        if is_quota_error(exc):
            error_text = (
                "تم تجاوز حصة Gemini Embeddings المجانية. "
                "انتظر دقيقة ثم أعد رفع الملف، أو فعّل الفوترة "
                "لرفع الحد.\n\n"
                f"التفاصيل: {error_text}"
            )

        await status.edit_text(f"❌ تعذر تحليل الملف:\n{error_text}")
    finally:
        file_path.unlink(missing_ok=True)


async def handle_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    يستقبل الرسائل الصوتية وملفات Audio من تيليجرام،
    يحولها إلى نص، ثم يضيف النص إلى Chroma كمصدر جديد.
    """
    message = update.effective_message
    if not message:
        return

    telegram_audio = message.voice or message.audio
    if not telegram_audio:
        return

    storage_id = user_storage_id(update)

    if message.voice:
        original_name = f"voice_{message.message_id}.ogg"
        mime_type = message.voice.mime_type or "audio/ogg"
        file_size = message.voice.file_size or 0
    else:
        original_name = (
            Path(message.audio.file_name).name
            if message.audio.file_name
            else f"audio_{message.message_id}.mp3"
        )
        mime_type = message.audio.mime_type or "audio/mpeg"
        file_size = message.audio.file_size or 0

    size_mb = file_size / (1024 * 1024)
    if size_mb > MAX_AUDIO_SIZE_MB:
        await message.reply_text(
            f"❌ حجم الملف الصوتي أكبر من الحد المسموح "
            f"({MAX_AUDIO_SIZE_MB} MB)."
        )
        return

    suffix = Path(original_name).suffix
    if not suffix:
        suffix = ".ogg" if message.voice else ".mp3"

    audio_path = user_download_dir(storage_id) / (
        f"{uuid.uuid4().hex}{suffix}"
    )

    status = await message.reply_text(
        "🎧 جاري تنزيل التسجيل وتحويله إلى نص..."
    )

    try:
        telegram_file = await telegram_audio.get_file()
        await telegram_file.download_to_drive(
            custom_path=audio_path,
        )

        detected_mime = detect_audio_mime_type(
            audio_path,
            mime_type,
        )

        transcript = await asyncio.to_thread(
            transcribe_audio_file,
            audio_path,
            detected_mime,
        )

        docs = [
            Document(
                page_content=transcript,
                metadata={
                    "source": original_name,
                    "source_type": "audio",
                    "mime_type": detected_mime,
                },
            )
        ]

        chunk_count = await asyncio.to_thread(
            add_documents,
            storage_id,
            docs,
        )

        await status.edit_text(
            "✅ تم تحويل التسجيل إلى نص وإضافته إلى المصادر.\n"
            f"عدد المقاطع النصية الجديدة: {chunk_count}"
        )

        await send_long_text(
            update,
            f"📝 التفريغ النصي:\n\n{transcript}",
        )

    except Exception as exc:
        logger.exception("فشل تحليل الملف الصوتي")
        error_text = str(exc).strip()
        if len(error_text) > 1500:
            error_text = error_text[:1500] + "..."

        await status.edit_text(
            f"❌ تعذر تحويل التسجيل إلى نص:\n{error_text}"
        )
    finally:
        audio_path.unlink(missing_ok=True)


async def handle_url(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if not message or not message.text:
        return

    storage_id = user_storage_id(update)
    url = message.text.strip()
    status = await message.reply_text("🌐 جاري قراءة الرابط...")

    try:
        if is_youtube_url(url):
            video_id = extract_youtube_id(url)
            if not video_id:
                raise ValueError("تعذر استخراج معرف فيديو YouTube.")

            transcript = await asyncio.to_thread(
                fetch_youtube_transcript,
                video_id,
            )

            if not transcript:
                raise ValueError("لم يتم العثور على نص للفيديو.")

            docs = [
                Document(
                    page_content=transcript,
                    metadata={
                        "source": url,
                        "source_type": "youtube",
                        "video_id": video_id,
                    },
                )
            ]

            chunk_count = await asyncio.to_thread(
                add_documents,
                storage_id,
                docs,
            )

            await status.edit_text(
                "✅ تمت إضافة نص فيديو YouTube إلى المصادر.\n"
                f"عدد المقاطع الجديدة: {chunk_count}"
            )
            return

        title, text = await asyncio.to_thread(fetch_web_page, url)

        docs = [
            Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title": title,
                    "source_type": "web",
                },
            )
        ]

        chunk_count = await asyncio.to_thread(
            add_documents,
            storage_id,
            docs,
        )

        await status.edit_text(
            f"✅ تمت إضافة صفحة الويب: {title}\n"
            f"عدد المقاطع الجديدة: {chunk_count}"
        )

    except requests.HTTPError as exc:
        logger.warning("HTTP error: %s", exc)
        await status.edit_text(
            f"❌ رفض الموقع الطلب أو أعاد خطأ HTTP: {exc.response.status_code}"
        )
    except Exception as exc:
        logger.exception("فشل معالجة الرابط")
        await status.edit_text(f"❌ تعذر إضافة الرابط: {exc}")


# =========================================================
# الأسئلة والإجابات
# =========================================================

async def answer_from_sources(
    storage_id: str,
    user_query: str,
) -> tuple[str, list[Document]]:
    store = get_vector_store(storage_id)

    selected_source = selected_source_for(storage_id)
    search_kwargs = {}
    if selected_source:
        search_kwargs["filter"] = {"source": selected_source}

    docs = await asyncio.to_thread(
        store.similarity_search,
        user_query,
        RETRIEVAL_K,
        **search_kwargs,
    )

    if not docs:
        return "لم أجد معلومات مرتبطة بالسؤال داخل المصادر.", []

    context_text = "\n\n---\n\n".join(
        f"[المصدر: {doc.metadata.get('source', 'غير معروف')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

    system_text = (
        "أنت مساعد بحث يعتمد على مصادر المستخدم فقط. "
        "أجب باللغة العربية الواضحة والمنظمة. "
        "لا تضف حقائق من معرفتك العامة إذا لم تكن موجودة في السياق. "
        "إذا كانت المعلومات غير كافية، صرّح بذلك بوضوح. "
        "عند وجود تعارض بين المصادر، اذكر التعارض. "
        "لا تقل إنك قرأت ملفًا كاملًا إذا كان السياق يحتوي على مقتطفات فقط.\n\n"
        f"السياق المسترجع:\n{context_text}"
    )

    messages = [SystemMessage(content=system_text)]
    messages.extend(history_for(storage_id))
    messages.append(HumanMessage(content=user_query))

    result = await llm.ainvoke(messages)
    answer = str(result.content).strip()

    history = history_for(storage_id)
    history.extend(
        [
            HumanMessage(content=user_query),
            AIMessage(content=answer),
        ]
    )
    trim_history(storage_id)

    return answer, docs


async def answer_from_sources_deep(
    storage_id: str,
    user_query: str,
    status_message=None,
):
    store = get_vector_store(storage_id)
    selected_source = selected_source_for(storage_id)

    async def progress(text: str) -> None:
        if status_message:
            await status_message.edit_text(text)

    result = await deep_source_engine.retrieve(
        store=store,
        query=user_query,
        selected_source=selected_source,
        history_text=short_history_text(storage_id),
        progress_callback=progress,
    )

    history = history_for(storage_id)
    history.extend(
        [
            HumanMessage(content=user_query),
            AIMessage(content=result.answer),
        ]
    )
    trim_history(storage_id)

    return result


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if not message or not message.text:
        return

    user_query = message.text.strip()

    if user_query.startswith(("http://", "https://")):
        await handle_url(update, context)
        return

    storage_id = user_storage_id(update)
    started = time.perf_counter()

    # التلخيص الكامل يبقى مسارًا خاصًا.
    if is_full_summary_request(user_query):
        if not database_has_documents(storage_id):
            await message.reply_text(
                "⚠️ لا توجد مصادر نصية بعد. أرسل ملف PDF أو رابطًا أو صوتًا أولًا."
            )
            return

        status = await message.reply_text(
            "📚 جاري تلخيص جميع المصادر على مراحل..."
        )
        try:
            summary, docs = await create_full_staged_summary(storage_id, status)
            await status.delete()
            answer = f"📘 الملخص الكامل للمصادر:\n\n{summary}"
            await send_long_text(update, answer)

            from research_core import QueryDecision
            decision = QueryDecision(
                original_query=user_query,
                rewritten_query=user_query,
                intent="full_summary",
                route="staged_summary",
            )
            await asyncio.to_thread(
                research_platform.log_interaction,
                storage_id,
                decision,
                response_time_ms=(time.perf_counter() - started) * 1000,
                retrieved_count=len(docs),
                answer=answer,
            )
        except Exception as exc:
            logger.exception("فشل التلخيص الكامل")
            await status.edit_text(f"❌ تعذر إنشاء التلخيص الكامل:\n{exc}")
        return

    decision = await research_platform.process_query(storage_id, user_query)

    if decision.route == "clarification":
        answer = decision.clarification_question or "يرجى توضيح المساق."
        await message.reply_text(answer)
        await asyncio.to_thread(
            research_platform.log_interaction,
            storage_id,
            decision,
            response_time_ms=(time.perf_counter() - started) * 1000,
            retrieved_count=0,
            answer=answer,
        )
        return

    if decision.route == "sql" and decision.structured_answer:
        suggestions = (
            "\n\n💡 يمكنك أيضًا السؤال عن المتطلبات السابقة، "
            "المحاضر، القاعة، أو محتوى المساق."
        )
        answer = decision.structured_answer + suggestions
        await send_long_text(update, answer)
        await asyncio.to_thread(
            research_platform.log_interaction,
            storage_id,
            decision,
            response_time_ms=(time.perf_counter() - started) * 1000,
            retrieved_count=0,
            answer=answer,
        )
        return

    if not database_has_documents(storage_id):
        await message.reply_text(
            "⚠️ لا توجد مصادر نصية للإجابة التفصيلية. "
            "أرسل ملف PDF أو رابطًا، أو اسأل عن بيانات المساقات المنظمة."
        )
        return

    use_deep_mode = deep_mode_enabled(storage_id)
    status = await message.reply_text(
        "🧠 بدأ Deep Source Mode..."
        if use_deep_mode
        else "⚡ جاري البحث المركز داخل المصادر..."
    )

    try:
        if use_deep_mode:
            deep_result = await answer_from_sources_deep(
                storage_id,
                decision.rewritten_query,
                status,
            )
            answer = deep_result.answer
            docs = deep_result.documents
            decision.route = "deep_rag"
        else:
            answer, docs = await answer_from_sources(
                storage_id,
                decision.rewritten_query,
            )
            deep_result = None
            decision.route = "rag"

        await status.delete()

        sources_text = build_sources_text(docs)
        final_text = answer

        if decision.rewritten_query != decision.original_query:
            final_text += (
                "\n\n🔄 فهمت السؤال في سياقه على أنه:\n"
                f"{decision.rewritten_query}"
            )

        if deep_result:
            sufficiency = (
                "كافية"
                if deep_result.sufficient
                else "جزئية أو غير مكتملة"
            )
            final_text += (
                "\n\n🧪 فحص الأدلة:"
                f"\n• الحالة: {sufficiency}"
                f"\n• الثقة: {deep_result.confidence:.2f}"
                f"\n• مراحل الاسترجاع: {deep_result.stages}"
                f"\n• المرشحون المفحوصون: "
                f"{deep_result.candidate_count}"
            )
            if deep_result.missing_evidence:
                final_text += (
                    "\n• الدليل الناقص: "
                    f"{deep_result.missing_evidence}"
                )

            if deep_result.evidence_labels:
                final_text += (
                    "\n\n🔖 خريطة علامات الأدلة:\n"
                    + "\n".join(deep_result.evidence_labels)
                )

        if sources_text:
            final_text += (
                f"\n\n📚 المصادر المسترجعة:\n{sources_text}"
            )

        final_text += (
            "\n\n💡 استخدم /deep للتحليل العميق، "
            "أو /precise للإجابة الأسرع."
        )

        await send_long_text(update, final_text)

        log_id = await asyncio.to_thread(
            research_platform.log_interaction,
            storage_id,
            decision,
            response_time_ms=(time.perf_counter() - started) * 1000,
            retrieved_count=len(docs),
            answer=answer,
        )

        if deep_result:
            await asyncio.to_thread(
                research_platform.log_deep_retrieval,
                research_log_id=log_id,
                session_id=storage_id,
                query_variants=deep_result.query_variants,
                stages=deep_result.stages,
                candidate_count=deep_result.candidate_count,
                evidence_count=len(deep_result.evidence_labels),
                sufficient=deep_result.sufficient,
                confidence=deep_result.confidence,
                missing_evidence=deep_result.missing_evidence,
            )

    except Exception as exc:
        logger.exception("فشل توليد الإجابة")
        error_text = str(exc).strip()
        if len(error_text) > 1200:
            error_text = error_text[:1200] + "..."
        await status.edit_text(
            f"❌ حدث خطأ أثناء البحث أو توليد الإجابة:\n{error_text}"
        )


# =========================================================
# البودكاست
# =========================================================

async def generate_podcast(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    storage_id = user_storage_id(update)

    if not database_has_documents(storage_id):
        await update.effective_message.reply_text(
            "⚠️ أضف مصادر أولًا قبل إنشاء البودكاست."
        )
        return

    status = await update.effective_message.reply_text(
        "🎙️ جاري إعداد ملخص حواري صوتي..."
    )

    audio_path = user_download_dir(storage_id) / (
        f"podcast_{uuid.uuid4().hex}.mp3"
    )

    try:
        store = get_vector_store(storage_id)
        selected_source = selected_source_for(storage_id)
        search_kwargs = {}
        if selected_source:
            search_kwargs["filter"] = {"source": selected_source}

        docs = await asyncio.to_thread(
            store.similarity_search,
            "الأفكار الأساسية والنتائج والمفاهيم الأكثر أهمية",
            10,
            **search_kwargs,
        )

        context_text = "\n\n".join(doc.page_content for doc in docs)
        prompt = (
            "اكتب سيناريو بودكاست عربي موجز وطبيعي بين أحمد ونور، "
            "اعتمادًا حصريًا على النص التالي. "
            "ليشرحا أهم الأفكار ويتبادلا الأسئلة والإجابات. "
            "لا تضف معلومات غير موجودة. "
            "اجعل المدة التقريبية 3 إلى 5 دقائق، "
            "ولا تستخدم Markdown أو رموزًا زخرفية.\n\n"
            f"{context_text[:16000]}"
        )

        result = await llm.ainvoke(prompt)
        script = str(result.content).strip()

        if not script:
            raise ValueError("لم يتم إنشاء نص للبودكاست.")

        clean_audio_text = re.sub(
            r"(?m)^\s*(أحمد|نور)\s*:\s*",
            "",
            script,
        )
        clean_audio_text = clean_audio_text.replace("**", "")

        await asyncio.to_thread(
            gTTS(
                text=clean_audio_text,
                lang="ar",
                slow=False,
            ).save,
            str(audio_path),
        )

        await status.edit_text("✅ تم إعداد البودكاست.")

        # نرسل النص أولًا على أجزاء
        await send_long_text(
            update,
            f"📝 سيناريو البودكاست:\n\n{script}",
        )

        with audio_path.open("rb") as audio_file:
            await update.effective_message.reply_audio(
                audio=audio_file,
                caption="🎙️ ملخص صوتي لمصادرك",
            )

    except Exception as exc:
        logger.exception("فشل إنشاء البودكاست")
        await status.edit_text(f"❌ تعذر إنشاء البودكاست: {exc}")
    finally:
        audio_path.unlink(missing_ok=True)


# =========================================================
# معالج الأخطاء والتشغيل
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    logger.exception(
        "خطأ غير معالج أثناء معالجة تحديث تيليجرام",
        exc_info=context.error,
    )


def main() -> None:
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("sources", sources_command))
    application.add_handler(CommandHandler("deep", deep_mode_command))
    application.add_handler(CommandHandler("precise", precise_mode_command))
    application.add_handler(CommandHandler("mode", retrieval_mode_command))
    application.add_handler(CommandHandler("courses", courses_command))
    application.add_handler(CommandHandler("seed_demo", seed_demo_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    application.add_handler(CommandHandler("research_stats", research_stats_command))
    application.add_handler(CommandHandler("export_logs", export_logs_command))
    application.add_handler(CommandHandler("files", files_command))
    application.add_handler(CommandHandler("select", select_source_command))
    application.add_handler(CommandHandler("all", all_sources_command))
    application.add_handler(CommandHandler("delete", delete_source_command))
    application.add_handler(CommandHandler("summary", full_summary_command))
    application.add_handler(CommandHandler("reset", reset_notebook))
    application.add_handler(CommandHandler("podcast", generate_podcast))

    application.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO,
            handle_audio,
        )
    )

    application.add_handler(
        MessageHandler(filters.Document.PDF, handle_document)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    application.add_error_handler(error_handler)

    logger.info("تم تشغيل البوت باستخدام Google Gemini: %s", CHAT_MODEL)
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )


if __name__ == "__main__":
    main()
