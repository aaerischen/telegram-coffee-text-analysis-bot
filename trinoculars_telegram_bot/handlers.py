from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext

from .config import MIN_TEXT_LENGTH, MODEL_TYPE, USE_BINOCULARS
from .model_service import (
    is_model_ready,
    classify_user_text,
    analyze_user_text,
    format_classification_result,
    format_stats,
)

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="ℹПомощь")]],
        resize_keyboard=True,
        input_field_placeholder="Отправьте текст для проверки",
    )

    text = (
        "<b>Добро пожаловать в Trinoculars‑бот!</b>\n\n"
        "Я использую модель Trinoculars для определения того, написан ли текст человеком "
        "или сгенерирован ИИ.\n\n"
        f"<b>Текущий тип модели:</b> {MODEL_TYPE}\n"
        f"<b>Минимальная длина текста:</b> {MIN_TEXT_LENGTH} символов.\n\n"
        "Просто отправьте текст сообщением, чтобы получить результат."
    )

    await message.answer(text, reply_markup=keyboard, parse_mode="HTML")


@router.message(Command("help"))
async def cmd_help(message: Message):
    text = (
        "<b>Справка по использованию бота</b>\n\n"
        "<b>Как пользоваться:</b>\n"
        f"Отправьте текст длиной не менее {MIN_TEXT_LENGTH} символов.\n"
        "Лучше всего работают достаточно длинные текстовые фрагменты (от 500 символов).\n\n"
        "<b>Команды:</b>\n"
        "/start — перезапустить бота и показать приветствие\n"
        "/help — показать эту справку\n"
        "/stats — получить детальный анализ текста\n"
        "/about — информация о Trinoculars\n"
    )
    await message.answer(text, parse_mode="HTML")


@router.message(Command("about"))
async def cmd_about(message: Message):
    text = (
        "<b>О проекте Trinoculars</b>\n\n"
        "Trinoculars — это модель классификации текстов (человек / ИИ / дополнительные классы),\n"
        "использующая лингвистические признаки для детекции машинной генерации.\n\n"
        "Данный бот позволяет производить быструю проверку текстов прямо в мессенджере."
    )
    await message.answer(text, parse_mode="HTML")


@router.message(Command("stats"))
async def cmd_stats(message: Message, state: FSMContext):
    await message.answer(
        "<b>Детальная статистика текста</b>\n\n"
        "Отправьте текст, для которого вы хотите получить подробный анализ.\n"
        "После ответа вы снова сможете отправлять тексты на обычную классификацию.",
        parse_mode="HTML",
    )
    await state.set_state("waiting_for_stats_text")


@router.message(lambda m: m.text and m.text.strip() == "Помощь")
async def help_button(message: Message):
    await cmd_help(message)


@router.message(lambda m: m.text and len(m.text.strip()) < MIN_TEXT_LENGTH)
async def handle_short_text(message: Message):
    txt = message.text.strip()
    await message.answer(
        "Текст слишком короткий для анализа.\n\n"
        f"Минимальная длина: {MIN_TEXT_LENGTH} символов.\n"
        f"Сейчас: {len(txt)}.\n\n"
        "Отправьте более длинный текст либо используйте /help."
    )


@router.message(lambda m: m.text and len(m.text.strip()) >= MIN_TEXT_LENGTH)
async def handle_text(message: Message, state: FSMContext):
    if not is_model_ready():
        await message.answer(
            "Модель ещё не загружена или произошла ошибка при инициализации.\n"
            "Пожалуйста, обратитесь к администратору."
        )
        return

    user_text = message.text.strip()

    current_state = await state.get_state()
    if current_state == "waiting_for_stats_text":
        await state.clear()
        waiting_msg = await message.answer("Анализирую текст...")
        try:
            analysis = analyze_user_text(user_text)
            formatted = format_stats(analysis)
            await waiting_msg.edit_text(formatted, parse_mode="HTML")
        except Exception as e:
            await waiting_msg.edit_text(
                f"Ошибка при анализе текста: {e}\n"
                "Попробуйте ещё раз или обратитесь к администратору."
            )
        return

    processing_msg = await message.answer("Анализирую текст... Пожалуйста, подождите.")

    try:
        result = classify_user_text(user_text, use_scores=USE_BINOCULARS)
        formatted = format_classification_result(result)
        await processing_msg.edit_text(formatted, parse_mode="HTML")
    except Exception as e:
        await processing_msg.edit_text(
            "Произошла ошибка при анализе текста.\n"
            f"Детали: {e}\n\n"
            "Попробуйте ещё раз или обратитесь к администратору."
        )


@router.message()
async def handle_other_messages(message: Message):
    if message.text:
        await message.answer(
            "Я умею работать только с текстовыми сообщениями.\n"
            f"Отправьте текст длиной не менее {MIN_TEXT_LENGTH} символов "
            "или используйте /help."
        )
    else:
        await message.answer(
            "Пожалуйста, отправьте текстовое сообщение. "
            "Я не могу анализировать этот тип содержимого."
        )


