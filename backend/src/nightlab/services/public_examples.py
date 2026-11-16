"""Bounded, explicit desktop imports of the already approved public PSG cohort.

Nothing is downloaded or trained. Catalog availability is only a cheap presence
check; each import verifies the official raw checksums and frozen preparation
receipt before reading any audio. No performance-based sample selection.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import soundfile as sf

from nightlab.audio.reference import TARGET_TYPES


EXAMPLES = {
    'psg-validation-1008': '00001008-100507',
    'psg-validation-1020': '00001020-100507',
}
CLIP_SECONDS = 120
LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/'
DATASET_URL = 'https://doi.org/10.11922/sciencedb.00345'


class PublicExampleService:
    def __init__(self, project_root, records):
        self.root = Path(project_root).resolve()
        self.records = records

    def _context(self, example_id):
        if example_id not in EXAMPLES:
            raise KeyError(example_id)
        from scripts.datasets.cli import load_manifest
        from scripts.datasets.prepare import output_path
        manifest = load_manifest(self.root/'data/manifests/psg-audio-pilot-10.json', self.root)
        corpus_path = output_path(self.root, 'data/processed/psg-audio-pilot-10/corpus.json')
        corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
        record_id = EXAMPLES[example_id]
        subject = record_id.split('-')[0]
        if (not corpus.get('complete') or corpus['dataset'] != manifest['id']
                or corpus['version'] != manifest['version']
                or corpus['split_policy'] != manifest['subjects']
                or subject not in manifest['subjects']['subjects']['validation']):
            raise ValueError('公开展示必须使用冻结验证集，且已完成准备与受试者划分')
        selected = [r for r in corpus['records'] if r['record_id'] == record_id and r.get('ok')]
        if len(selected) != 1:
            raise ValueError('公开展示记录不完整或重复')
        return manifest, selected[0]

    def list(self):
        from scripts.datasets.cli import raw_path
        from scripts.datasets.prepare import output_path
        result = []
        for example_id, record_id in EXAMPLES.items():
            available = False
            try:
                manifest, record = self._context(example_id)
                entries = [f for f in manifest['files'] if f.get('record_id') == record_id]
                available = (len(entries) == 2 and all(raw_path(self.root, f['path']).is_file() for f in entries)
                             and (output_path(self.root, record['path'])/'metadata.json').is_file())
            except (OSError, ValueError, KeyError, TypeError):
                pass
            result.append(dict(id=example_id, name=f'PSG-Audio {record_id.split("-")[0]} · 前两分钟',
                               available=available, duration_seconds=CLIP_SECONDS, split='validation',
                               reason=None if available else '需先按文档校验并准备公开十人数据；不会自动下载',
                               scope='固定验证集片段；PSG呼吸参考，不是逐声标签或新的正式评估'))
        return result

    def import_example(self, example_id):
        from scripts.datasets.dense_audio import record_context
        manifest, selected = self._context(example_id)
        _, metadata, recording, events, offset, provenance = record_context(self.root, manifest, selected)
        if recording.duration_seconds < CLIP_SECONDS or metadata['channel'] != 'Mic':
            raise ValueError('公开样例要求至少两分钟的已验收环境麦克风通道')
        rate = recording.sample_rate('Mic')
        if int(rate) != rate:
            raise ValueError('公开样例采样率必须为整数')
        wave = recording.read_window('Mic', 0, CLIP_SECONDS)
        references = []
        for event in events:
            if event.kind not in TARGET_TYPES or event.start >= offset+CLIP_SECONDS or event.end <= offset:
                continue
            references.append(dict(start_seconds=max(0., event.start-offset),
                                   end_seconds=min(float(CLIP_SECONDS), event.end-offset),
                                   label=event.kind, source='reference', score=None, model_id=None,
                                   reference_clipped=event.start < offset or event.end > offset+CLIP_SECONDS))
        source = dict(kind='public_dataset', dataset=manifest['id'], version=manifest['version'],
                      record_id=selected['record_id'], subject_id=provenance['subject_id'],
                      split='validation', segment_start_seconds=0, duration_seconds=CLIP_SECONDS,
                      rml_offset_seconds=offset, channel='Mic', dataset_url=DATASET_URL,
                      license='CC BY 4.0', license_url=LICENSE_URL,
                      attribution='Korompili et al., PSG-Audio V3; Scientific Data 8, 197 (2021), doi:10.1038/s41597-021-00977-w',
                      input_sha256=provenance['input_sha256'],
                      reference_basis='RML PSG respiratory intervals, clipped to the fixed excerpt',
                      time_basis='relative to EDF segment; original clock has no verified timezone',
                      audio_conversion='EDF fixed ADC full scale to float32 WAV; no loudness normalization',
                      reference_count=len(references))
        path = self.records.staging_dir/(str(uuid4())+'.wav')
        try:
            sf.write(path, wave, int(rate), subtype='FLOAT', format='WAV')
            return self.records.import_audio(path, f'PSG-Audio {provenance["subject_id"]} · 公开验证片段',
                                             consent=True, provenance=source, reference_events=references)
        finally:
            if path.exists():
                self.records._safe(path, self.records.staging_dir).unlink()
