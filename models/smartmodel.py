from .faceX import FaceTrainingWrapper
from .classifier import VisionWrapper


def get_model(model_cfg, logger, rank):
    assert 'task' in model_cfg, 'Task is not specified'

    match model_cfg['task']:
        case 'face' | 'cbir': return FaceTrainingWrapper(model_cfg, logger)
        case 'classification': return VisionWrapper(model_cfg, logger, rank)