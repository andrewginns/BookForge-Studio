"""
Step: Generate OpenAI TTS Audio

Generates audio using the OpenAI TTS microservice.
"""

from backend.core.data_types.step_context import StepContext
from backend.core.utils.generation_utils import generate_candidates_with_service_calls

# Import logging utilities
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


async def process(context: StepContext):
    """
    Generate multiple audio candidates using OpenAI TTS microservice.

    Args:
        context: StepContext containing text array and parameters
    """
    sentences = context.multiple_speaker_text_array or []

    if not sentences:
        raise Exception("No text array available for audio generation")

    logger.info(f"Starting OpenAI TTS audio generation for {len(sentences)} sentences")
    logger.info(f"Sentences: {sentences}")
    logger.info(f"Voice clone paths: {context.voice_clone_paths}")
    logger.info(f"Transcriptions: {context.audio_transcriptions}")
    logger.info(f"Generating {context.num_candidates} candidates")

    await generate_candidates_with_service_calls(
        model_name="openai_tts",
        context=context,
        texts=sentences,
        voice_paths=context.voice_clone_paths,
        audio_transcriptions=context.audio_transcriptions
        or ["" for _ in context.voice_clone_paths],
    )


# Step metadata
STEP_METADATA = {
    "name": "generate_openai_tts_audio",
    "display_name": "Generate OpenAI TTS Audio",
    "description": "Generates audio using OpenAI TTS model",
    "input_type": "string",
    "output_type": "audio",
    "category": "audio-generation",
    "step_type": "generation_step",
    "version": "1.0.0",
    "model_requirement": "openai_tts",
}
