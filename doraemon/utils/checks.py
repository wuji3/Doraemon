from functools import reduce
from pathlib import Path
import os
from datasets import load_dataset

def check_cfgs_common(cfgs):
    def find_normalize(augment_list):
        for augment in augment_list:
            if 'normalize' in augment:
                return augment['normalize']
        return None

    hyp_cfg = cfgs['hyp']
    data_cfg = cfgs['data']
    model_cfg = cfgs['model']

    # Check loss configuration
    assert reduce(lambda x, y: int(x) + int(y[0]), list(hyp_cfg['loss'].values())) == 1, \
        'Loss configuration error: Only one loss type should be enabled. Set either ce: true or bce: [true, ...] in hyp.loss'

    # Check optimizer
    assert hyp_cfg['optimizer'][0] in {'sgd', 'adam', 'sam'}, \
        'Invalid optimizer selection. Please choose from: sgd, adam, or sam'

    # Check scheduler and warm-up settings
    valid_schedulers = {'linear', 'cosine', 'linear_with_warm', 'cosine_with_warm'}
    assert hyp_cfg['scheduler'] in valid_schedulers, \
        'Invalid scheduler selection. Supported options: linear, cosine, linear_with_warm, cosine_with_warm'

    assert hyp_cfg['warm_ep'] >= 0 and isinstance(hyp_cfg['warm_ep'], int) and hyp_cfg['warm_ep'] < hyp_cfg['epochs'], \
        f'Invalid warm-up epochs: must be a non-negative integer less than total epochs ({hyp_cfg["epochs"]})'

    if hyp_cfg['warm_ep'] == 0:
        assert hyp_cfg['scheduler'] in {'linear', 'cosine'}, \
            'When warm-up is disabled (warm_ep: 0), only linear or cosine scheduler is supported'
    if hyp_cfg['warm_ep'] > 0:
        assert hyp_cfg['scheduler'] in {'linear_with_warm', 'cosine_with_warm'}, \
            'When using warm-up (warm_ep > 0), scheduler must be either linear_with_warm or cosine_with_warm'

    # Check normalization settings
    train_normalize = find_normalize(data_cfg['train']['augment'])
    val_normalize = find_normalize(data_cfg['val']['augment'])

    # Check backbone configuration
    if 'backbone' in model_cfg:
        backbone_cfg = next(iter(model_cfg['backbone'].items()))
        backbone_name, backbone_params = backbone_cfg
    else:
        backbone_name = model_cfg['name']
        backbone_params = {
            'pretrained': model_cfg.get('pretrained', False),
            'image_size': model_cfg.get('image_size')
        }

    # Verify model type is timm
    assert backbone_name.startswith('timm-'), \
        "Only timm models are supported. Model name must start with 'timm-'"

    # Check normalization requirements based on pretrained status
    is_pretrained = backbone_params.get('pretrained', False)
    if is_pretrained:
        if train_normalize is None or val_normalize is None:
            raise ValueError('Pretrained models require normalization in both training and validation augmentations')
        if train_normalize['mean'] != val_normalize['mean'] or train_normalize['std'] != val_normalize['std']:
            raise ValueError('Inconsistent normalization parameters: mean and std must be identical for training and validation')
    
    # Check image size for backbone
    assert 'image_size' in backbone_params, \
        f'Image size must be specified for {backbone_name}'
    assert backbone_params['image_size'] == model_cfg['image_size'], \
        f'Image size mismatch: {backbone_params["image_size"]} in backbone config vs {model_cfg["image_size"]} in model config'

def check_cfgs_face(cfgs):
    """
    Check configurations specific to face recognition tasks.
    
    Args:
        cfgs: Configuration dictionary
    """
    check_cfgs_common(cfgs=cfgs)

    model_cfg = cfgs['model']
    data_cfg = cfgs['data']

        # Determine data source type
    is_local = os.path.isdir(data_cfg['root'])

    # Check number of classes based on data source
    if is_local:
        train_classes = [x for x in os.listdir(Path(data_cfg['root'])/'train') 
                        if not (x.startswith('.') or x.startswith('_'))]
        num_classes = len(train_classes)
    else:
        try:
            dataset = load_dataset(data_cfg['root'], split='train')
            num_classes = len(set(dataset['label']))
        except Exception as e:
            raise ValueError(f"Dataset loading error: Unable to load HuggingFace dataset from {data_cfg['root']}. Details: {str(e)}")

    # Check model configuration
    head_key = next(iter(model_cfg['head'].keys()))
    model_classes = model_cfg['head'][head_key]['num_class']
    
    assert model_classes == num_classes, \
        f'Model configuration error: Number of classes mismatch. Expected {num_classes} from dataset, but got {model_classes} in model configuration'

    # # Check face recognition specific configurations
    # if cfgs['model']['task'] == 'face':
    #     pair_txt_path = data_cfg['val']['pair_txt']
    #     
    #     # Verify pair text file existence
    #     if not os.path.isfile(pair_txt_path):
    #         raise ValueError(f'Validation data error: Pair text file not found at {pair_txt_path}')

    #     # Validate pair list format
    #     from doraemon.engine.representation.eval_face import Evaluator
    #     try:
    #         with open(pair_txt_path) as f:
    #             pair_list = [line.strip() for line in f.readlines()]
    #         Evaluator.check_nps(pair_list)
    #     except Exception as e:
    #         raise ValueError(f'Pair list validation error: Invalid format in {pair_txt_path}. Details: {str(e)}')

def check_cfgs_cbir(cfgs):
    """
    Check configurations specific to CBIR (Content-Based Image Retrieval) tasks.
    
    Args:
        cfgs: Configuration dictionary
    """
    check_cfgs_common(cfgs=cfgs)

    model_cfg = cfgs['model']
    data_cfg = cfgs['data']

    # Determine data source type
    is_local = os.path.isdir(data_cfg['root'])

    # Check number of classes based on data source
    if is_local:
        train_classes = [x for x in os.listdir(Path(data_cfg['root'])/'train') 
                        if not (x.startswith('.') or x.startswith('_'))]
        num_classes = len(train_classes)
    else:
        try:
            dataset = load_dataset(data_cfg['root'], split='train')
            num_classes = len(set(dataset['label']))
        except Exception as e:
            raise ValueError(f"Dataset loading error: Unable to load HuggingFace dataset from {data_cfg['root']}. Details: {str(e)}")

    # Check model configuration
    head_key = next(iter(model_cfg['head'].keys()))
    model_classes = model_cfg['head'][head_key]['num_class']
    
    assert model_classes == num_classes, \
        f'Model configuration error: Number of classes mismatch. Expected {num_classes} from dataset, but got {model_classes} in model configuration'

def check_cfgs_classification(cfgs):
    """
    Check configurations specific to classification tasks.
    
    Args:
        cfgs: Configuration dictionary
    """
    check_cfgs_common(cfgs=cfgs)

    model_cfg = cfgs['model']
    data_cfg = cfgs['data']
    hyp_cfg = cfgs['hyp']

    # Determine data source type
    is_csv = data_cfg['root'].endswith('.csv')
    is_local = os.path.isdir(data_cfg['root'])

    # Check loss configuration based on data source
    if is_csv:
        if hyp_cfg['loss']['ce']:
            raise ValueError('Loss configuration error: Multi-label tasks (CSV format) require BCE loss. Please set ce: false in hyp.loss')
        if not hyp_cfg['loss']['bce'][0]:
            raise ValueError('Loss configuration error: Multi-label tasks (CSV format) require BCE loss. Please set bce: [true, ...] in hyp.loss')
    else:
        if not hyp_cfg['loss']['ce']:
            raise ValueError('Loss configuration error: Single-label tasks (folder structure/HuggingFace) require CE loss. Please set ce: true in hyp.loss')
        if hyp_cfg['loss']['bce'][0]:
            raise ValueError('Loss configuration error: Single-label tasks (folder structure/HuggingFace) do not support BCE loss. Please set bce: [false, ...] in hyp.loss')

    # Check num_classes
    if is_local:
        train_classes = [x for x in os.listdir(Path(data_cfg['root'])/'train') 
                        if not (x.startswith('.') or x.startswith('_'))]
        num_classes = len(train_classes)
    elif is_csv:
        import pandas as pd
        df = pd.read_csv(data_cfg['root'])
        class_columns = [col for col in df.columns if col not in ['image_path', 'train']]
        num_classes = len(class_columns)
        if hyp_cfg['loss']['bce'][0]:
            assert num_classes == len(hyp_cfg['loss']['bce'][1]), \
                f'Loss configuration error: Number of classes mismatch. Expected {len(hyp_cfg["loss"]["bce"][1])} from dataset, but got {num_classes} in model configuration'
    else:
        try:
            dataset = load_dataset(data_cfg['root'], split='train')
            num_classes = len(set(dataset['label']))
        except Exception as e:
            raise ValueError(f"Dataset loading error: Unable to load HuggingFace dataset from {data_cfg['root']}. Details: {str(e)}")

    assert model_cfg['num_classes'] == num_classes, \
        f'Model configuration error: Number of classes mismatch. Expected {num_classes} from dataset, but got {model_cfg["num_classes"]} in model configuration'

    # Check model configuration
    assert model_cfg['name'].split('-')[0] == 'timm', \
        'Model name error: Format should be [timm-ModelName] for timm models'

    if model_cfg['kwargs'] and model_cfg['pretrained']:
        for k in model_cfg['kwargs'].keys():
            if k not in {'dropout', 'attention_dropout', 'stochastic_depth_prob'}:
                raise KeyError('Model kwargs error: When using pretrained models, only [dropout, attention_dropout, stochastic_depth_prob] are allowed')

    # Check training strategies
    if hyp_cfg['strategy']['focal'][0]:
        assert hyp_cfg['loss']['bce'], \
            'Strategy configuration error: Focal loss requires BCE loss. Please enable BCE loss'
    
    if hyp_cfg['strategy']['ohem'][0]:
        assert not hyp_cfg['loss']['bce'][0], \
            'Strategy configuration error: OHEM is not compatible with BCE loss. Please disable BCE loss'

    # Check mixup configuration
    mix_ratio, mix_duration = hyp_cfg['strategy']['mixup']["ratio"], hyp_cfg['strategy']['mixup']["duration"]
    
    # Basic ratio check
    assert 0 <= mix_ratio <= 1, 'Mixup configuration error: ratio must be in [0,1]'
    
    # Only check duration when mixup is enabled
    if mix_ratio > 0:
        assert 0 < mix_duration <= hyp_cfg['epochs'], \
            f'Mixup configuration error: when mixup is enabled (ratio > 0), duration must be in (0,{hyp_cfg["epochs"]}]'
    
    hyp_cfg['strategy']['mixup'] = [mix_ratio, mix_duration]

def check(task, cfgs):
    if task == 'face': check_cfgs_face(cfgs)
    elif task == 'cbir': check_cfgs_cbir(cfgs)
    elif task == 'classification': check_cfgs_classification(cfgs)
    else: raise ValueError(f'{task} is not supported')