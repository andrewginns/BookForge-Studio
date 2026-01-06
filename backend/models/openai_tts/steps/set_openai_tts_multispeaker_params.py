"""
Step: Set OpenAI TTS Multi-Speaker Parameters

This step sets parameters for multi-speaker OpenAI TTS generation.
"""


async def process(context):
    """
    Set OpenAI TTS multi-speaker parameters.

    Args:
        context: StepContext to set parameters
    """
    if "response_format" not in context.parameters:
        context.parameters["response_format"] = "wav"

    if "instructions" not in context.parameters:
        context.parameters["instructions"] = "Speak in a natural, expressive tone."


STEP_METADATA = {
    "name": "set_openai_tts_multispeaker_params",
    "display_name": "Set OpenAI TTS Multi-Speaker Parameters",
    "description": "Sets parameters for multi-speaker OpenAI TTS generation",
    "input_type": "parameter",
    "output_type": "parameter",
    "category": "parameter-setting",
    "step_type": "start_step",
    "multi-speaker": True,
    "version": "1.0.0",
    "model_requirement": "openai_tts",
    "parameters": {
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
