from sfind.config.config import Context
from sfind.model_engine.frontend.captioning.captioning_model_frontend import CaptioningModelFrontEnd
from sfind.model_engine.frontend.captioning.sales_force_blip_base_frontend import SalesForceBlipBaseModelFrontEnd
from sfind.model_engine.frontend.encoder.clip_model_frontend import CLIPModelFrontend
from sfind.model_engine.frontend.encoder.encoder_model_frontend import EncoderModelFrontEnd
from sfind.model_engine.frontend.encoder.perception_encoder_frontend import PerceptionEncoderModelFrontend


class ModelFactory:
    @staticmethod
    def get_encoder_model(context: Context) -> EncoderModelFrontEnd:
        model_type = context.config.model_config.encoder_model_type
        if model_type == "pe":
            return PerceptionEncoderModelFrontend()
        elif model_type == "clip":
            return CLIPModelFrontend()
        else:
            raise ValueError(f"Unknown encoder model frontend: {model_type}")

    @staticmethod
    def get_captioning_model(context: Context) -> CaptioningModelFrontEnd:
        model_type = context.config.model_config.captioning_model_type
        if model_type == "sf-blip-base":
            return SalesForceBlipBaseModelFrontEnd()
        else:
            raise ValueError(f"Unknown captioning model frontend: {model_type}")
