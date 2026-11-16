"""Explicit, bounded publication of aggregate classic-run metrics to the local UI."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

MAX_BYTES = 2 * 1024 * 1024
NAME = re.compile(r'[a-z0-9][a-z0-9_-]{0,79}')
SPLITS = {'train', 'validation', 'test'}
DATASETS = {'psg-audio-pilot-10', 'sleepaccel-cohort-31', 'sleepaccel-pilot-3'}


def safe_name(name):
    if not isinstance(name, str) or not NAME.fullmatch(name) or name.split('.')[0].upper() in {
        'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('run name must be a safe single portable name')
    return name


def owned(root, *parts):
    root = Path(root).resolve()
    path = root
    for part in parts:
        if part in {'.', '..'} or '/' in part or '\\' in part:
            raise ValueError('path traversal rejected')
        path = path / part
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError('linked experiment paths rejected')
    if not path.resolve().is_relative_to(root):
        raise ValueError('experiment path escapes project')
    return path


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('missing or oversized aggregate metrics')
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('oversized aggregate metrics')
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON')))
        def check(item, depth=0):
            if depth > 20:
                raise ValueError('JSON nesting too deep')
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError('non-finite number')
            if isinstance(item, dict):
                for child in item.values(): check(child, depth + 1)
            elif isinstance(item, list):
                for child in item: check(child, depth + 1)
        check(value)
    except (RecursionError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid aggregate JSON') from exc
    return value, hashlib.sha256(raw).hexdigest()


def integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= 10**9:
        raise ValueError('invalid aggregate count')
    return value


def score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid aggregate score')
    return value


def aggregate(source):
    """Allowlist output: never expose provenance paths, identities, or arbitrary strings."""
    if not isinstance(source, dict) or source.get('dataset') not in DATASETS:
        raise ValueError('unsupported classic dataset/report')
    labels = ['W', 'N1', 'N2', 'N3', 'REM'] if source['dataset'].startswith('sleepaccel') else ['no_target_reference_event', 'reference_respiratory_event']
    models = source.get('models')
    if not isinstance(models, dict) or set(models) != {'majority', 'random_forest'}:
        raise ValueError('expected classic majority and random_forest metrics')
    clean = {}
    for name, partitions in models.items():
        if not isinstance(partitions, dict) or set(partitions) != {'validation', 'test'}:
            raise ValueError('expected held-out validation/test metrics')
        clean[name] = {}
        for split, metrics in partitions.items():
            if not isinstance(metrics, dict) or metrics.get('label_order') != labels:
                raise ValueError('unexpected metric label order')
            allowed = {'macro_f1', 'balanced_accuracy', 'accuracy', 'label_order', 'confusion_matrix', 'per_class', 'absent_reference_classes', 'balanced_accuracy_scope', 'macro_f1_scope', 'pr_auc_average_precision', 'pr_auc_undefined_reason'}
            if set(metrics) - allowed:
                raise ValueError('unexpected nonaggregate metric fields')
            result = {key: score(metrics[key]) for key in ('macro_f1', 'balanced_accuracy', 'accuracy')}
            matrix = metrics['confusion_matrix']
            if not isinstance(matrix, list) or len(matrix) != len(labels) or any(not isinstance(row, list) or len(row) != len(labels) for row in matrix):
                raise ValueError('invalid confusion matrix shape')
            result.update(label_order=labels, confusion_matrix=[[integer(n) for n in row] for row in matrix])
            per_class = metrics['per_class']
            if not isinstance(per_class, dict) or set(per_class) != set(labels):
                raise ValueError('invalid per-class metrics')
            result['per_class'] = {}
            for i, label in enumerate(labels):
                values = per_class[label]
                if not isinstance(values, dict) or set(values) != {'precision', 'recall', 'f1', 'support'}:
                    raise ValueError('invalid class aggregate')
                result['per_class'][label] = {key: score(values[key]) for key in ('precision', 'recall', 'f1')}
                support = integer(values['support'])
                if support != sum(result['confusion_matrix'][i]):
                    raise ValueError('class support differs from confusion matrix')
                result['per_class'][label]['support'] = support
            result['absent_reference_classes'] = [label for label in labels if result['per_class'][label]['support'] == 0]
            result['macro_f1_scope'] = 'fixed declared class list; absent classes contribute zero'
            result['balanced_accuracy_scope'] = 'mean recall over classes present in this reference split'
            if 'pr_auc_average_precision' in metrics:
                result['pr_auc_average_precision'] = None if metrics['pr_auc_average_precision'] is None else score(metrics['pr_auc_average_precision'])
                if result['pr_auc_average_precision'] is None:
                    result['pr_auc_undefined_reason'] = 'reference split must contain both positive and negative classes'
            clean[name][split] = result
    config = source.get('config')
    if not isinstance(config, dict) or set(config) != {'seed', 'trees', 'max_depth', 'min_samples_leaf'}:
        raise ValueError('invalid classic config')
    config = {key: integer(value, 0 if key == 'seed' else 1) for key, value in config.items()}
    rows = source.get('split_rows')
    groups = source.get('split_subjects')
    if not isinstance(rows, dict) or set(rows) != SPLITS or not isinstance(groups, dict) or set(groups) != SPLITS:
        raise ValueError('missing split metadata')
    rows = {key: integer(value, 1) for key, value in rows.items()}
    identities = []
    for values in groups.values():
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or len(v) > 128 for v in values):
            raise ValueError('invalid subject partition')
        identities.extend(values)
    if len(set(identities)) != len(identities) or integer(source.get('subject_count'), 1) != len(identities):
        raise ValueError('subject leakage or count mismatch')
    for split in ('validation', 'test'):
        for model in clean.values():
            if sum(map(sum, model[split]['confusion_matrix'])) != rows[split]:
                raise ValueError('split row mismatch')
    subset = source.get('feature_subset', 'all')
    if subset not in {'all', 'heart_rate', 'acceleration'}:
        raise ValueError('invalid feature subset')
    return {'dataset': source['dataset'], 'models': clean, 'config': config, 'feature_subset': subset,
            'feature_count': integer(source.get('feature_count'), 1), 'subject_count': len(identities),
            'split_subject_counts': {k: len(v) for k, v in groups.items()}, 'split_rows': rows,
            'evaluation_unit': 'held-out public-data windows/epochs; non-clinical research'}


def register(root, run_name, execute=False):
    name = safe_name(run_name)
    source, digest = read_json(owned(root, 'runs', name, 'metrics.json'))
    report = aggregate(source)
    target = owned(root, 'private', 'local-experiments', name + '.json')
    if target.exists():
        raise ValueError('local experiment already registered; refusing overwrite')
    snapshot = {'schema': 1, 'run_name': name, 'source_metrics_sha256': digest, 'report': report}
    if execute:
        target.parent.mkdir(parents=True, exist_ok=True)
        target = owned(root, 'private', 'local-experiments', name + '.json')
        fd, temporary = tempfile.mkstemp(prefix='.register-', suffix='.tmp', dir=target.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-link publication is atomic and refuses a concurrent existing target.
            os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {'mode': 'registered' if execute else 'dry-run', 'id': 'local_' + name, 'report': report,
            'source_metrics_sha256': digest}


def detail(root, name):
    name = safe_name(name)
    snapshot, _ = read_json(owned(root, 'private', 'local-experiments', name + '.json'))
    if not isinstance(snapshot, dict) or set(snapshot) != {'schema', 'run_name', 'source_metrics_sha256', 'report'} or snapshot['schema'] != 1 or snapshot['run_name'] != name or not re.fullmatch('[0-9a-f]{64}', str(snapshot['source_metrics_sha256'])):
        raise ValueError('invalid registered snapshot')
    # Revalidate the snapshot via the same allowlist, supplying anonymous split placeholders.
    report = snapshot['report']
    if not isinstance(report, dict) or set(report) != {'dataset', 'models', 'config', 'feature_subset', 'feature_count', 'subject_count', 'split_subject_counts', 'split_rows', 'evaluation_unit'}:
        raise ValueError('invalid aggregate snapshot')
    counts = report['split_subject_counts']
    if not isinstance(counts, dict) or set(counts) != SPLITS:
        raise ValueError('invalid subject counts')
    groups = {key: [f'{key}_{i}' for i in range(integer(n, 1))] for key, n in counts.items() if integer(n, 1) <= 1000}
    clean = aggregate({**report, 'split_subjects': groups})
    return {'id': 'local_' + name, 'title': f'本地复现实验 · {name}', 'description': '显式登记的本机经典 ML 聚合快照',
            'category': '本地复现', 'dataset': clean['dataset'], 'origin': 'local', 'available': True, 'figures': [],
            'report': {**clean, 'source_metrics_sha256': snapshot['source_metrics_sha256']}}


def unavailable(identity):
    """A fixed readable failure card; never echo a path, filename, or exception."""
    safe_id = identity if re.fullmatch(r'local_invalid_[0-9a-f]{16}', identity) else 'local_invalid_' + hashlib.sha256(identity.encode('utf-8', errors='replace')).hexdigest()[:16]
    message = '本地登记报告校验失败或不可读取；请检查本机登记文件并重新登记。其他实验仍可查看。'
    return {'id': safe_id, 'title': '本地登记报告不可用', 'description': message,
            'category': '本地复现', 'origin': 'local', 'available': False, 'figures': [],
            'report': {'registration_status': 'unavailable', 'message': message}}
