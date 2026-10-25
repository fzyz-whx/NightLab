"""Small deterministic pilot training with train-only moments and validation selection."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import os
import platform
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from nightlab.deep.inputs import MaskedChannelStandardizer
from nightlab.deep.models import build_model
from nightlab.experiments.baseline import classification_metrics
from nightlab.experiments.splits import validate_subject_assignments


LABELS = {'audio':['no_target_reference_event','reference_respiratory_event'],
          'wearable':['W','N1','N2','N3','REM']}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 20
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = .001
    weight_decay: float = .0001
    gradient_clip_norm: float = 5.0
    cpu_threads: int = 2

    def __post_init__(self):
        for key in ('seed','epochs','patience','batch_size','cpu_threads'):
            value = getattr(self,key)
            if type(value) is not int or value < (0 if key == 'seed' else 1):
                raise ValueError(f'invalid {key}')
        if self.cpu_threads > 2 or not np.isfinite([self.learning_rate,self.weight_decay,self.gradient_clip_norm]).all():
            raise ValueError('invalid thread or optimizer settings')
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError('invalid optimizer settings')


def select_device(requested='cpu'):
    if requested not in {'cpu','cuda','auto'}:
        raise ValueError('device must be cpu/cuda/auto')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable; explicitly choose CPU or auto')
    return torch.device('cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu')


def prepare_tensors(task, values, masks, train_indices, feature_set='all'):
    """Select modality before fitting; absent observations become zero plus mask."""
    values = np.asarray(values)
    if task not in LABELS or feature_set not in {'all','heart_rate','acceleration'}:
        raise ValueError('unknown task or modality')
    if task == 'audio':
        if feature_set != 'all' or values.ndim != 3 or values.shape[1:] != (32,100) or not np.isfinite(values).all():
            raise ValueError('audio requires finite N x 32 x 100 log-mel images')
        selected = values
        selected_masks = None
    else:
        masks = np.asarray(masks)
        if values.ndim != 3 or values.shape[1:] != (5,30) or masks.shape != (len(values),2,30):
            raise ValueError('wearable requires N x 5 x 30 values and N x 2 x 30 masks')
        observed = masks[:,[0,1,1,1,1],:].astype(bool)
        if (not np.isin(masks,(0,1)).all() or not np.isfinite(values[observed]).all()
                or not np.isnan(values[~observed]).all()):
            raise ValueError('wearable masks and missing values disagree')
        columns,mask_columns = {'all':([0,1,2,3,4],[0,1]),'heart_rate':([0],[0]),
                                'acceleration':([1,2,3,4],[1])}[feature_set]
        selected,selected_masks = values[:,columns,:],masks[:,mask_columns,:]
    normalizer = MaskedChannelStandardizer().fit(selected[train_indices])
    normalized = normalizer.transform(selected)
    x = normalized[:,None,:,:] if task == 'audio' else np.concatenate((normalized,selected_masks),axis=1)
    return torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)), {
        'mean':normalizer.mean_.tolist(), 'scale':normalizer.scale_.tolist(),
        'fit_scope':'supervised training subject rows only; observed finite values',
        'fit_rows':len(train_indices), 'feature_set':feature_set,
        'missing_fill':'standardized zero with independent observation mask (wearable only)'}


def predict_probabilities(model, features, device, batch_size=64):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for begin in range(0,len(features),batch_size):
            logits = model(features[begin:begin+batch_size].to(device))
            probabilities.append(torch.softmax(logits,dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def fit_pilot(task, values, masks, targets, subjects, supervised_mask, assignments,
              feature_set='all', device='cpu', config=None, smoke=False, progress=None):
    config = config or TrainingConfig()
    targets,subjects,supervised_mask = np.asarray(targets),np.asarray(subjects),np.asarray(supervised_mask)
    if (targets.ndim != 1 or subjects.shape != targets.shape or supervised_mask.shape != targets.shape
            or len(targets) != len(values) or supervised_mask.dtype != np.bool_):
        raise ValueError('input identities, labels and supervised mask differ')
    labels = LABELS[task]
    if not set(targets[supervised_mask]).issubset(labels) or np.any(targets[~supervised_mask] != ''):
        raise ValueError('unknown or fabricated supervised targets')
    validate_subject_assignments(assignments,subjects)
    indices = {name:np.flatnonzero(np.isin(subjects,group)&supervised_mask) for name,group in assignments.items()}
    if any(not len(index) for index in indices.values()) or set(targets[indices['train']]) != set(labels):
        raise ValueError('nonempty sets and all declared training classes required')
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    selected_device = select_device(device)
    if selected_device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(selected_device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    features,normalizer = prepare_tensors(task,values,masks,indices['train'],feature_set)
    model = build_model(task, in_channels=features.shape[1]).to(selected_device)
    encoded = np.array([labels.index(label) if valid else -1 for label,valid in zip(targets,supervised_mask,strict=True)])
    counts = np.bincount(encoded[indices['train']],minlength=len(labels))
    weights = len(indices['train'])/(len(labels)*counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=selected_device))
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    history,best_state,best_epoch,best_score,stale = [],None,0,-1.0,0
    started = time.perf_counter()
    for epoch in range(1,(1 if smoke else config.epochs)+1):
        model.train()
        order = indices['train'][torch.randperm(len(indices['train']),generator=generator).numpy()]
        loss_sum,seen = 0.0,0
        for batch_index,begin in enumerate(range(0,len(order),config.batch_size)):
            if smoke and batch_index >= 5:
                break
            rows = order[begin:begin+config.batch_size]
            batch = features[rows].to(selected_device)
            truth = torch.tensor(encoded[rows],dtype=torch.long,device=selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch),truth)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm,error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())*len(rows); seen += len(rows)
        probabilities = predict_probabilities(model,features[indices['validation']],selected_device,config.batch_size)
        validation = classification_metrics(targets[indices['validation']],np.asarray(labels)[probabilities.argmax(axis=1)],labels,
                                             probabilities[:,1] if task == 'audio' else None,
                                             labels[1] if task == 'audio' else None)
        item = {'epoch':epoch, 'training_loss_batch_weighted':loss_sum/seen, 'training_rows_seen':seen,
                'validation_macro_f1':validation['macro_f1'], 'elapsed_seconds':time.perf_counter()-started}
        history.append(item)
        if validation['macro_f1'] > best_score:
            best_score,best_epoch,stale = validation['macro_f1'],epoch,0
            best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:
            stale += 1
        if progress:
            progress(item)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    reports,outputs = {},{}
    for part in (('validation',) if smoke else ('validation','test')):
        rows = np.flatnonzero(np.isin(subjects,assignments[part]))
        probabilities = predict_probabilities(model,features[rows],selected_device,config.batch_size)
        valid = supervised_mask[rows]
        reports[part] = classification_metrics(targets[rows][valid],np.asarray(labels)[probabilities[valid].argmax(axis=1)],labels,
                                               probabilities[valid,1] if task == 'audio' else None,
                                               labels[1] if task == 'audio' else None)
        outputs[part] = {'rows':rows, 'probabilities':probabilities}
    # weights_only deserialization is verified by the CLI against these predictions.
    report = {'task':task, 'feature_set':feature_set, 'config':asdict(config), 'seed':config.seed,
              'selection':'highest validation Macro-F1; strict improvement, earliest tie, patience stop; test never selects',
              'best_epoch':best_epoch, 'best_validation_macro_f1':best_score, 'history':history,
              'class_weights_training_only':weights.tolist(), 'training_class_counts':counts.tolist(),
              'normalizer':normalizer, 'label_order':labels, 'split_subjects':assignments,
              'split_rows':{name:len(rows) for name,rows in indices.items()},
              'parameter_count':sum(parameter.numel() for parameter in model.parameters()),
              'models':{'cnn':reports}, 'device':str(selected_device),
              'gpu_name':torch.cuda.get_device_name(selected_device) if selected_device.type == 'cuda' else None,
              'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(selected_device) if selected_device.type == 'cuda' else 0,
              'elapsed_seconds':time.perf_counter()-started, 'smoke_only':smoke,
              'environment':{'python':platform.python_version(), **{name:version(name) for name in
                              ('torch','numpy','scipy','scikit-learn','nightlab')}},
              'scope':'pilot held-out subjects; no population, clinical or device-generalization validation'}
    return best_state,report,outputs,features
