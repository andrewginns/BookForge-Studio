"""
Step: Set OpenAI TTS Single-Speaker Parameters

This step sets parameters for single-speaker OpenAI TTS generation.
"""


async def process(context):
    """
    Set OpenAI TTS single-speaker parameters.

    Args:
        context: StepContext to set parameters
    """
    if "voice" not in context.parameters:
        context.parameters["voice"] = "marin"

    if "response_format" not in context.parameters:
        context.parameters["response_format"] = "wav"

    if "instructions" not in context.parameters:
        context.parameters["instructions"] = "Speak in a natural, expressive tone."


STEP_METADATA = {
    "name": "set_openai_tts_singlespeaker_params",
    "display_name": "Set OpenAI TTS Single-Speaker Parameters",
    "description": "Sets parameters for single-speaker OpenAI TTS generation",
    "input_type": "parameter",
    "output_type": "parameter",
    "category": "parameter-setting",
    "step_type": "start_step",
    "multi-speaker": False,
    "version": "1.0.0",
    "model_requirement": "openai_tts",
    "parameters": {
        "voice": {
            "type": "string",
            "default": "marin",
            "description": "Preset OpenAI voice name",
        },
        "response_format": {
            "type": "string",
            "default": "wav",
            "description": "Audio output format (wav recommended)",
        },
        "instructions": {
            "type": "string",
            "default": "Speak in a natural, expressive tone.",
            "description": "Style guidance for the voice",
        },
    },
}
