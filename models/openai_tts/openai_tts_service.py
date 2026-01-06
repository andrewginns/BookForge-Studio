#!/usr/bin/env python3
"""
OpenAI TTS Microservice

FastAPI service for OpenAI text-to-speech using preset voices.
Provides ElevenLabs-compatible endpoints.
"""

import hashlib
import logging
import os
import tempfile
from typing import Dict, List, Optional, Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException

# Import shared models
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_models import (
    DialogueInput,
    TextToSpeechRequest,
    TextToDialogueRequest,
    GenerationResponse,
    LoadModelResponse,
    UnloadModelResponse,
    ServiceInfoResponse,
    OpenAIModel,
    OpenAIModelsResponse,
)

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from backend.core.utils.url_utils import decode_filepath_from_url

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError as e:
    print(f"Error importing librosa: {e}")
    LIBROSA_AVAILABLE = False

from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenAI TTS Service", version="1.0.0")

MODEL_NAME = "OpenAI TTS"
OPENAI_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts-2025-12-15")
DEFAULT_VOICE = os.getenv("OPENAI_TTS_DEFAULT_VOICE", "marin")

AVAILABLE_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="OPENAI_API_KEY is required for OpenAI TTS"
        )
    return OpenAI(api_key=api_key)


def normalize_voice_name(voice_value: Optional[str]) -> Optional[str]:
    if not voice_value:
        return None

    if voice_value in AVAILABLE_VOICES:
        return voice_value

    try:
        decoded = decode_filepath_from_url(voice_value)
    except Exception:
        decoded = voice_value

    decoded = os.path.basename(decoded)
    decoded = os.path.splitext(decoded)[0]
    if decoded in AVAILABLE_VOICES:
        return decoded
    return None


def select_voice(voice_key: Optional[str], fallback_index: int = 0) -> str:
    normalized = normalize_voice_name(voice_key)
    if normalized:
        return normalized

    if voice_key:
        digest = hashlib.sha256(voice_key.encode("utf-8")).digest()
        hashed_index = int.from_bytes(digest[:4], "big") % len(AVAILABLE_VOICES)
        return AVAILABLE_VOICES[hashed_index]

    return AVAILABLE_VOICES[fallback_index % len(AVAILABLE_VOICES)]


def build_instructions(
    parameters: Dict[str, Any],
    transcription: Optional[str] = None,
) -> str:
    instructions = parameters.get(
        "instructions", "Speak in a natural, expressive tone."
    )
    if transcription and parameters.get("use_transcription_guidance", True):
        instructions = f"{instructions} Style reference: {transcription}"
    return instructions


def generate_audio_to_file(
    client: OpenAI,
    text: str,
    voice: str,
    output_path: str,
    instructions: str,
    response_format: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_MODEL,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format=response_format,
    ) as response:
        response.stream_to_file(output_path)


def concatenate_audio_files(audio_paths: List[str], output_path: str) -> None:
    audio_segments = []
    target_sr = None

    for path in audio_paths:
        audio, sr = sf.read(path)
        if target_sr is None:
            target_sr = sr
        elif sr != target_sr:
            if not LIBROSA_AVAILABLE:
                raise HTTPException(
                    status_code=500,
                    detail="Audio sample rates differ and librosa is unavailable for resampling.",
                )
            if audio.ndim == 1:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            else:
                channels = []
                for channel in audio.T:
                    channels.append(
                        librosa.resample(channel, orig_sr=sr, target_sr=target_sr)
                    )
                audio = np.stack(channels, axis=1)
        audio_segments.append(audio)

    if not audio_segments:
        raise HTTPException(status_code=500, detail="No audio segments to concatenate.")

    combined = np.concatenate(audio_segments, axis=0)
    sf.write(output_path, combined, target_sr)


@app.get("/", response_model=ServiceInfoResponse)
async def root():
    return ServiceInfoResponse(message=MODEL_NAME, model=OPENAI_MODEL)


@app.post("/v1/text-to-speech/{voice_id}", response_model=GenerationResponse)
async def text_to_speech(voice_id: str, request: TextToSpeechRequest):
    try:
        client = get_openai_client()
        output_path = request.output_filepath
        parameters = request.parameters or {}
        voice_override = normalize_voice_name(parameters.get("voice"))
        voice = voice_override or select_voice(voice_id)
        transcription = (
            request.audio_transcriptions[0]
            if request.audio_transcriptions
            else None
        )
        instructions = build_instructions(parameters, transcription)
        response_format = parameters.get("response_format", "wav")

        generate_audio_to_file(
            client=client,
            text=request.text,
            voice=voice,
            output_path=output_path,
            instructions=instructions,
            response_format=response_format,
        )

        return GenerationResponse(
            status="success",
            output_filepath=output_path,
            message="OpenAI TTS generation completed",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/text-to-dialogue", response_model=GenerationResponse)
async def text_to_dialogue(request: TextToDialogueRequest):
    try:
        client = get_openai_client()
        output_path = request.output_filepath
        parameters = request.parameters or {}
        response_format = parameters.get("response_format", "wav")
        if response_format != "wav":
            logger.warning(
                "Multi-speaker OpenAI TTS enforces WAV for concatenation; overriding response_format."
            )
            response_format = "wav"

        voice_map: Dict[str, str] = {}
        voice_override = normalize_voice_name(parameters.get("voice"))
        temp_files: List[str] = []
        temp_dir = tempfile.mkdtemp()

        for idx, inp in enumerate(request.inputs):
            voice_key = inp.voice_id
            if voice_key not in voice_map:
                voice_map[voice_key] = (
                    voice_override
                    or select_voice(voice_key, fallback_index=idx)
                )
            voice = voice_map[voice_key]
            transcription = (
                request.audio_transcriptions[idx]
                if request.audio_transcriptions and idx < len(request.audio_transcriptions)
                else None
            )
            instructions = build_instructions(parameters, transcription)
            temp_path = os.path.join(temp_dir, f"segment_{idx}.wav")
            generate_audio_to_file(
                client=client,
                text=inp.text,
                voice=voice,
                output_path=temp_path,
                instructions=instructions,
                response_format=response_format,
            )
            temp_files.append(temp_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        concatenate_audio_files(temp_files, output_path)

        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        return GenerationResponse(
            status="success",
            output_filepath=output_path,
            message="OpenAI TTS dialogue generation completed",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI TTS dialogue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/load-model", response_model=LoadModelResponse)
async def load_model():
    return LoadModelResponse(
        status="success", message="OpenAI TTS does not require local model loading"
    )


@app.post("/v1/unload-model", response_model=UnloadModelResponse)
async def unload_model_endpoint():
    return UnloadModelResponse(
        status="success", message="OpenAI TTS does not require local model unloading"
    )


@app.get("/v1/models", response_model=OpenAIModelsResponse)
async def list_models():
    import time

    model_data = OpenAIModel(
        id=OPENAI_MODEL,
        created=int(time.time()),
        owned_by="openai",
        status="healthy",
        model_loaded=True,
        device="remote",
        dependencies_available=True,
    )
    return OpenAIModelsResponse(data=[model_data])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8007)
