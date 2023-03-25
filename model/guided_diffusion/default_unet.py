from .script_util import create_model_and_diffusion
from .config import model_and_diffusion_defaults


def get_unet():
    config = model_and_diffusion_defaults()
    model, _ = create_model_and_diffusion(**config)
    return model
