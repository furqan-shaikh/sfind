import json
from pathlib import Path
from sfind.models.models import Context, Config, ModelConfig

CONFIG_PATH = Path(__file__).parent / "config.json"
def load_config():
    with open(CONFIG_PATH, 'r') as file:
        return json.load(file)


def create_context():
    config_json = load_config()
    return Context(
        config=Config(
            model_config=ModelConfig(
                encoder_model_type=config_json["model"]["encoder_frontend"],
                captioning_model_type=config_json["model"]["captioning_frontend"],
            )
        )
    )
