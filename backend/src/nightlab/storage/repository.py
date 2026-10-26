"""Small SQLite repository: connections never escape an operation."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def valid_id(value):
    if str(UUID(str(value))) != str(value):
        raise ValueError("无效的记录 UUID")
    return str(value)


class Repository:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "nightlab.sqlite3"
        with self._connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS wearable(record_id TEXT PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE, status TEXT NOT NULL, data TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_job ON jobs(record_id) WHERE status IN ('queued','running');
            ''')

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _record(db, record_id):
        valid_id(record_id)
        row = db.execute("SELECT data FROM records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        return json.loads(row[0])

    def list_records(self):
        with self._connection() as db:
            return sorted((json.loads(r[0]) for r in db.execute("SELECT data FROM records")), key=lambda r: r["created_at"], reverse=True)

    def get_record(self, record_id):
        with self._connection() as db:
            return self._record(db, record_id)

    def create_record(self, record):
        valid_id(record["id"])
        with self._connection() as db:
            db.execute("INSERT INTO records VALUES (?,?)", (record["id"], json.dumps(record, allow_nan=False)))
        return record

    def get_events(self, record_id):
        with self._connection() as db:
            self._record(db, record_id)
            return sorted((json.loads(r[0]) for r in db.execute("SELECT data FROM events WHERE record_id=?", (record_id,))), key=lambda e: (e["start_seconds"], e["end_seconds"]))

    @staticmethod
    def _insert_event(db, event):
        event = dict(event)
        event.setdefault("id", str(uuid4()))
        valid_id(event["id"])
        if event["source"] not in {"model", "user", "demo"}:
            raise ValueError("未知标记来源")
        if event["source"] != "model":
            event["score"] = event["model_id"] = None
        db.execute("INSERT INTO events VALUES (?,?,?)", (event["id"], event["record_id"], json.dumps(event, allow_nan=False)))
        return event

    def add_event(self, event):
        with self._connection() as db:
            self._record(db, event["record_id"])
            return self._insert_event(db, event)

    def delete_event(self, record_id, event_id):
        with self._connection() as db:
            self._record(db, record_id)
            row = db.execute("SELECT data FROM events WHERE id=? AND record_id=?", (event_id, record_id)).fetchone()
            if row is None:
                raise KeyError(event_id)
            if json.loads(row[0])["source"] == "model":
                raise ValueError("模型标记请通过重新分析替换")
            db.execute("DELETE FROM events WHERE id=?", (event_id,))

    def set_wearable(self, record_id, wearable):
        with self._connection() as db:
            self._record(db, record_id)
            db.execute("INSERT OR REPLACE INTO wearable VALUES (?,?)", (record_id, json.dumps(wearable, allow_nan=False)))
        return wearable

    def get_wearable(self, record_id):
        with self._connection() as db:
            self._record(db, record_id)
            row = db.execute("SELECT data FROM wearable WHERE record_id=?", (record_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def create_job(self, job):
        valid_id(job["id"])
        with self._connection() as db:
            self._record(db, job["record_id"])
            try:
                db.execute("INSERT INTO jobs VALUES (?,?,?,?)", (job["id"], job["record_id"], job["status"], json.dumps(job, allow_nan=False)))
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("该记录已有活动任务或任务标识重复") from exc
        return job

    @staticmethod
    def _job(db, job_id):
        valid_id(job_id)
        row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def get_job(self, job_id):
        with self._connection() as db:
            return self._job(db, job_id)

    def update_job(self, job_id, **fields):
        if {"id", "record_id"} & fields.keys():
            raise ValueError("任务标识不可修改")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = self._job(db, job_id)
            job.update(fields)
            db.execute("UPDATE jobs SET status=?,data=? WHERE id=?", (job["status"], json.dumps(job, allow_nan=False), job_id))
            return job

    def _find_job(self, record_id, active):
        with self._connection() as db:
            self._record(db, record_id)
            sql = "SELECT data FROM jobs WHERE record_id=?" + (" AND status IN ('queued','running')" if active else "")
            jobs = [json.loads(r[0]) for r in db.execute(sql, (record_id,))]
            return max(jobs, key=lambda j: j["created_at"]) if jobs else None

    def latest_job(self, record_id):
        return self._find_job(record_id, False)

    def active_job(self, record_id):
        return self._find_job(record_id, True)

    def recover_interrupted_jobs(self):
        """A new single-process worker cannot resume the previous process's jobs."""
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = list(db.execute("SELECT data FROM jobs WHERE status IN ('queued','running')"))
            for row in rows:
                job = json.loads(row[0])
                job.update(status="failed", message="上次分析因服务停止而中断，可重新分析",
                           error="服务重启，未发布部分结果", finished_at=utc_now())
                db.execute("UPDATE jobs SET status=?,data=? WHERE id=?",
                           (job["status"], json.dumps(job, allow_nan=False), job["id"]))
            return len(rows)

    def complete_analysis(self, job_id, events):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = self._job(db, job_id)
            if job["status"] not in {"queued", "running"}:
                raise RuntimeError("任务已经结束，不能发布结果")
            db.execute("DELETE FROM events WHERE record_id=? AND json_extract(data,'$.source')='model'", (job["record_id"],))
            for event in events:
                self._insert_event(db, {**event, "record_id": job["record_id"], "source": "model",
                                        "model_id": job["model_id"], "model_version": job.get("model_version")})
            job.update(status="completed", progress=1.0, finished_at=utc_now(), processed_windows=job["total_windows"])
            db.execute("UPDATE jobs SET status=?,data=? WHERE id=?", (job["status"], json.dumps(job, allow_nan=False), job_id))
            return job

    def delete_record(self, record_id, delete_files=None):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._record(db, record_id)
            if db.execute("SELECT 1 FROM jobs WHERE record_id=? AND status IN ('queued','running')", (record_id,)).fetchone():
                raise RuntimeError("请先取消活动分析任务，等待任务停止后再删除")
            if delete_files:
                delete_files()
            db.execute("DELETE FROM records WHERE id=?", (record_id,))
