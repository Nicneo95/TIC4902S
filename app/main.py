from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from groq import APIConnectionError, APIStatusError, AsyncGroq, RateLimitError
# Speech-to-text disabled
# from fastapi import File, Form, UploadFile

from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
)
# from app.schemas import TranscriptionResponse

# Speech-to-text limits and supported formats:
# MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024
# SUPPORTED_AUDIO_EXTENSIONS = {
#     ".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm",
# }

async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.groq_client = (
        AsyncGroq(api_key=settings.groq_api_key) if settings.groq_api_key else None
    )
    yield
    if app.state.groq_client is not None:
        await app.state.groq_client.close()


app = FastAPI(
    title="Wanderwise Itinerary Planner",
    description="Generate travel itineraries from typed requests.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    # QUIZ-TO-PROMPT CODE GOES HERE: if payload contains quiz answers,
    # turn them into a user/system message before calling Groq below.
    # Keep the normal payload.messages path for regular chat requests.
    client: Optional[AsyncGroq] = request.app.state.groq_client
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq is not configured. Set GROQ_API_KEY in your .env file.",
        )
    # add a system message to the beginning of the messages list
    # add quiz answer as prompt to the system message if quiz answers are present in the payload here
    # messages = [
    #     {
    #         "role": "system",
    #         "content": (
    #             "You are a helpful travel itinerary assistant. Create practical "
    #             "day-by-day itineraries. Include activity, food, and transport "
    #         ),
    #     }
    # ]

    # if payload.quiz:
    #     quiz = payload.quiz
    #     quiz_prompt = (
    #         "Create a practical day-by-day travel itinerary using these preferences:\n"
    #         f"Destination: {quiz.destination}\n"
    #         f"Trip length: {quiz.trip_days} days\n"
    #     )
    #     messages.append({"role": "user", "content": quiz_prompt})

    try:
        completion = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[message.model_dump() for message in payload.messages],
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="Groq rate limit exceeded.") from exc
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=502, detail="Could not connect to Groq. Please try again."
        ) from exc
    except APIStatusError as exc:
        raise HTTPException(
            status_code=502, detail="Groq returned an error while generating a response."
        ) from exc

    content = completion.choices[0].message.content or ""
    return ChatResponse(content=content, model=completion.model)


# Speech-to-text endpoint disabled
# @app.post("/transcribe", response_model=TranscriptionResponse)
# async def transcribe(
#     request: Request,
#     file: UploadFile = File(...),
#     language: Optional[str] = Form(default=None),
#     prompt: Optional[str] = Form(default=None),
# ) -> TranscriptionResponse:
#     client: Optional[AsyncGroq] = request.app.state.groq_client
#     if client is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Groq is not configured. Set GROQ_API_KEY in your .env file.",
#         )
#
#     filename = file.filename or ""
#     if Path(filename).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail="Unsupported audio file type. Use flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.",
#         )
#
#     audio = await file.read(MAX_AUDIO_SIZE_BYTES + 1)
#     await file.close()
#     if not audio:
#         raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
#     if len(audio) > MAX_AUDIO_SIZE_BYTES:
#         raise HTTPException(status_code=413, detail="Audio file must be 25 MB or smaller.")
#
#     try:
#         result = await client.audio.transcriptions.create(
#             file=(filename, audio),
#             model=settings.groq_transcription_model,
#             response_format="json",
#             language=language,
#             prompt=prompt,
#         )
#     except RateLimitError as exc:
#         raise HTTPException(status_code=429, detail="Groq rate limit exceeded.") from exc
#     except APIConnectionError as exc:
#         raise HTTPException(
#             status_code=502, detail="Could not connect to Groq. Please try again."
#         ) from exc
#     except APIStatusError as exc:
#         raise HTTPException(
#             status_code=502, detail="Groq returned an error while transcribing the audio."
#         ) from exc
#
#     return TranscriptionResponse(
#         text=result.text,
#         model=settings.groq_transcription_model,
#     )
