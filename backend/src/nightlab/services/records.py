"""Consent-gated private media lifecycle."""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from nightlab.storage.repository import Repository, utc_now, valid_id
from nightlab.services.media import audio_info_validate


class RecordService:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        self.repo = Repository(self.data_dir)
        self.staging_dir = self.data_dir / "staging"
        self.records_dir = self.data_dir / "records"
        for directory in (self.staging_dir, self.records_dir):
            if directory.is_symlink() or directory.resolve().parent != self.data_dir:
                raise ValueError("私有存储目录不能是链接或越出根目录")
            directory.mkdir(exist_ok=True)

    @staticmethod
    def _safe(path, root):
        path, root = Path(path), Path(root)
        if root.is_symlink() or root.resolve() != root:
            raise ValueError("存储根目录发生变化")
        # Test every component before following links (also Windows junctions).
        try:
            relative = path.absolute().relative_to(root)
        except ValueError as exc:
            raise ValueError("文件必须位于受控私有目录内") from exc
        cursor = root
        for part in relative.parts:
            if part in {"..", "."}:
                raise ValueError("禁止路径跳转")
            cursor = cursor / part
            if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
                raise ValueError("私有文件不能通过符号链接或目录联接访问")
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved == root:
            raise ValueError("文件路径越出受控目录")
        return resolved

    def import_audio(self, staging_path, name, start_time=None, consent=False, is_demo=False, stop_requested=None):
        stage = self._safe(Path(staging_path), self.staging_dir)
        destination = None
        created_directory = False
        try:
            if consent is not True:
                raise ValueError("请先确认对所有可能被录到的人拥有录音使用权")
            if start_time is not None:
                try:
                    parsed = datetime.fromisoformat(start_time)
                except (TypeError, ValueError) as exc:
                    raise ValueError("录音开始时间必须为带时区的ISO时间") from exc
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError("录音开始时间必须包含时区")
                start_time = parsed.isoformat()
            info = audio_info_validate(stage,stop_requested=stop_requested)
            if stop_requested and stop_requested():
                raise InterruptedError('音频导入已取消')
            record_id = str(uuid4())
            destination = self._safe(self.records_dir / record_id, self.records_dir)
            destination.mkdir()
            created_directory = True
            os.replace(stage, destination / "audio.wav")
            record = dict(id=record_id, name=str(name), created_at=utc_now(), start_time=start_time,
                          is_demo=bool(is_demo), consent=True, audio=info)
            self.repo.create_record(record)
            return record
        except BaseException:
            if created_directory and destination.exists():
                self._remove_tree(destination)
            raise
        finally:
            if stage.exists():
                self._safe(stage, self.staging_dir).unlink()

    def audio_path(self, record_id):
        valid_id(record_id)
        self.repo.get_record(record_id)
        path = self._safe(self.records_dir / record_id / "audio.wav", self.records_dir)
        if not path.is_file():
            raise FileNotFoundError("录音文件缺失，请删除此记录后重新导入")
        return path

    def detail(self, record_id):
        return dict(record=self.repo.get_record(record_id), events=self.repo.get_events(record_id),
                    wearable=self.repo.get_wearable(record_id), latest_job=self.repo.latest_job(record_id))

    def _remove_tree(self, directory):
        directory = self._safe(directory, self.records_dir)
        if directory.exists():
            # Preflight the whole tree before recursive deletion follows any component.
            for current, directories, files in os.walk(directory, followlinks=False):
                for name in directories + files:
                    self._safe(Path(current) / name, self.records_dir)
            shutil.rmtree(directory)

    def delete_record(self, record_id, confirmed=False):
        if confirmed is not True:
            raise ValueError("删除原始录音和所有附件前必须明确确认")
        valid_id(record_id)
        directory = self._safe(self.records_dir / record_id, self.records_dir)
        # BEGIN IMMEDIATE also excludes a racing job creation. DB removal happens
        # only after physical deletion; an IO failure leaves a retryable index.
        self.repo.delete_record(record_id, lambda: self._remove_tree(directory))
        return dict(deleted=True, id=record_id)
