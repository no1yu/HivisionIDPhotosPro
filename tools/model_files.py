from pathlib import Path

from tools.exceptions import ModelNotFoundError


MODELS_DIR = Path(__file__).resolve().parent.parent / "creator" / "models"


def get_model_files(folder, *file_names):
    model_directory = MODELS_DIR / folder
    model_files = tuple(model_directory / file_name for file_name in file_names)
    if not all(model_file.is_file() for model_file in model_files):
        raise ModelNotFoundError()
    return tuple(str(model_file) for model_file in model_files)
