import logging
import os
from typing import Dict, Optional, Tuple

from .config import TRINOCULARS_PATH, MODEL_TYPE, MODEL_DIR, USE_BINOCULARS

# Import Trinoculars modules (paths are prepared in config)
from model_utils import load_model, classify_text  # type: ignore
from text_analysis import analyze_text  # type: ignore
from binoculars_utils import (  # type: ignore
    initialize_binoculars,
    compute_scores as _compute_scores,
)

logger = logging.getLogger(__name__)

# Global model state (loaded once at startup)
_model = None
_scaler = None
_label_encoder = None
_imputer = None

_bino_chat = None
_bino_coder = None


def load_detection_model(
    model_type: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> bool:
    global _model, _scaler, _label_encoder, _imputer

    model_type = model_type or MODEL_TYPE
    model_dir = model_dir or MODEL_DIR

    logger.info(
        "Loading Trinoculars model (type=%s, dir=%s) from %s",
        model_type,
        model_dir,
        TRINOCULARS_PATH,
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(str(TRINOCULARS_PATH))
        _model, _scaler, _label_encoder, _imputer = load_model(
            model_type=model_type,
            model_dir=model_dir,
        )
        logger.info("Trinoculars model loaded successfully")

        # Optionally initialize Binoculars observers
        if USE_BINOCULARS:
            _initialize_binoculars_if_needed()

        return True
    except Exception as e:
        logger.exception("Error loading Trinoculars model: %s", e)
        _model = _scaler = _label_encoder = _imputer = None
        return False
    finally:
        os.chdir(original_cwd)


def is_model_ready() -> bool:
    return _model is not None


def _initialize_binoculars_if_needed() -> None:
    global _bino_chat, _bino_coder

    if _bino_chat is not None or _bino_coder is not None:
        return

    try:
        logger.info("Initializing Binoculars models (this may take a while)...")
        _bino_chat, _bino_coder = initialize_binoculars()
        logger.info("Binoculars initialized.")
    except Exception as e:
        logger.exception("Failed to initialize Binoculars: %s", e)
        _bino_chat = _bino_coder = None


def classify_user_text(
    text: str,
    use_scores: bool = False,
) -> Dict:
    if not is_model_ready():
        raise RuntimeError("Model is not loaded")

    scores: Optional[Dict] = None
    if use_scores:
        _initialize_binoculars_if_needed()
        if _bino_chat is not None or _bino_coder is not None:
            scores = _compute_scores(text, _bino_chat, _bino_coder)

    # We don't change CWD here because model_utils works with already loaded objects
    result = classify_text(
        text,
        model=_model,
        scaler=_scaler,
        label_encoder=_label_encoder,
        imputer=_imputer,
        scores=scores,
    )
    return result


def analyze_user_text(text: str) -> Dict:
    return analyze_text(text)


def format_classification_result(result: Dict) -> str:
    predicted_class = result["predicted_class"]
    probabilities = result["probabilities"]

    # Basic mapping for Russian UI; adjust class name checks if needed
    cls_str = str(predicted_class).lower()
    if "human" in cls_str or "человек" in cls_str:
        emoji = "✅"
        status = "ЧЕЛОВЕК"
        color_info = "Текст, вероятно, написан человеком"
    else:
        emoji = "🤖"
        status = "ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ"
        color_info = "Текст, вероятно, сгенерирован ИИ"

    message = f"{emoji} <b>Результат классификации</b>\n\n"
    message += f"<b>Предсказанный класс:</b> {status}\n\n"
    message += "<b>Вероятности классов:</b>\n"

    for cls, prob in probabilities.items():
        name = str(cls)
        percentage = prob * 100
        bar_len = int(percentage / 5)  # up to ~20 symbols
        bar = "█" * bar_len + "░" * (20 - bar_len)
        message += f"{name}: {percentage:.1f}% {bar}\n"

    message += f"\n<i>{color_info}</i>"
    return message


def format_stats(analysis: Dict) -> str:
    basic = analysis.get("basic_stats", {})
    diversity = analysis.get("lexical_diversity", {})
    structure = analysis.get("text_structure", {})
    readability = analysis.get("readability", {})

    text = "<b>Детальная статистика текста</b>\n\n"

    text += "<b>Основная статистика:</b>\n"
    text += f"Токенов: {basic.get('total_tokens', 0)}\n"
    text += f"Слов: {basic.get('total_words', 0)}\n"
    text += f"Уникальных слов: {basic.get('unique_words', 0)}\n"
    text += f"Стоп-слов: {basic.get('stop_words', 0)}\n"
    avg_len = basic.get("avg_word_length", 0.0)
    text += f"Средняя длина слова: {avg_len:.2f} символов\n\n"

    text += "<b>Лексическое разнообразие:</b>\n"
    ttr = diversity.get("ttr", 0.0)
    mtld = diversity.get("mtld", 0.0)
    text += f"TTR (отношение типов к токенам): {ttr:.3f}\n"
    text += f"MTLD (мера разнообразия): {mtld:.2f}\n\n"

    text += "<b>Структура текста:</b>\n"
    text += f"Предложений: {structure.get('sentence_count', 0)}\n"
    avg_sent = structure.get("avg_sentence_length", 0.0)
    text += f"Средняя длина предложения: {avg_sent:.2f} токенов\n\n"

    text += "<b>Читабельность:</b>\n"
    fk = readability.get("flesh_kincaid_score", 0.0)
    wps = readability.get("words_per_sentence", 0.0)
    text += f"Индекс Флеша-Кинкейда: {fk:.2f}\n"
    text += f"Слов на предложение: {wps:.2f}\n"

    return text


