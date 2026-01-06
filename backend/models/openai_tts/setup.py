from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


def provide_model_metadata():
    return {
        "name": "OpenAI TTS",
        "voice_clone_tips": [
            "OpenAI TTS uses preset voices; voice cloning is not available by default.",
            "Assign a preset voice or provide tone guidance via instructions.",
        ],
        "default_workflows": [
            {
                "name": "OpenAI TTS Single-Speaker Workflow",
                "steps": [
                    "set_openai_tts_singlespeaker_params",
                    "generate_openai_tts_audio",
                    "export_audio",
                ],
            },
            {
                "name": "OpenAI TTS Multi-Speaker Workflow",
                "steps": [
                    "set_openai_tts_multispeaker_params",
                    "generate_openai_tts_audio",
                    "export_audio",
                ],
            },
        ],
    }
