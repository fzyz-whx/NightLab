"""One local CPU worker, cooperative cancellation, atomic result publication."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import threading
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from nightlab.audio.events import decode_events
from nightlab.services.media import read_mono_window
from nightlab.storage.repository import utc_now


def audio_windows(frames,rate):
    length,hop = rate*10,rate*5
    if frames<length:
        raise ValueError('该模型需要至少10秒录音，可先浏览或添加用户标记')
    starts = list(range(0,frames-length+1,hop))
    if starts[-1]+length<frames:
        starts.append(frames-length)
    return [(start/rate,(start+length)/rate) for start in starts]


class JobManager:
    def __init__(self,service,registry,num_threads=2):
        self.service,self.repo,self.registry = service,service.repo,registry
        self.num_threads = min(2,max(1,int(num_threads)))
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='nightlab-local')
        self._lock = threading.RLock()
        self._flags,self._futures = {},{}
        self.repo.recover_interrupted_jobs()

    def submit(self,record_id,model_id):
        version = self.registry.require(model_id)
        record = self.repo.get_record(record_id)
        windows = audio_windows(record['audio']['frames'],record['audio']['sample_rate'])
        job = {'id':str(uuid4()),'record_id':record_id,'model_id':model_id,'model_version':version,
               'status':'queued','progress':0.0,'processed_windows':0,'total_windows':len(windows),
               'message':'等待本地分析','error':None,'created_at':utc_now(),'finished_at':None}
        with self._lock:
            self.repo.create_job(job)
            flag = threading.Event()
            self._flags[job['id']] = flag
            self._futures[job['id']] = self.executor.submit(self._work,job['id'],windows,flag)
        return job

    def _work(self,job_id,windows,flag):
        try:
            with self._lock:
                if flag.is_set():
                    self._cancelled(job_id); return
                job = self.repo.update_job(job_id,status='running',message='核验录音完整性')
            path = self.service.audio_path(job['record_id'])
            record = self.repo.get_record(job['record_id'])
            checksum = hashlib.sha256()
            with path.open('rb') as stream:
                while block:=stream.read(1024*1024):
                    if flag.is_set():
                        self._cancelled(job_id); return
                    checksum.update(block)
            if checksum.hexdigest()!=record['audio']['sha256']:
                raise ValueError('录音导入后发生变化，请重新导入')
            probabilities = []
            with threadpool_limits(limits=self.num_threads):
                for begin in range(0,len(windows),32):
                    waves = []
                    for start,end in windows[begin:begin+32]:
                        if flag.is_set():
                            self._cancelled(job_id); return
                        wave,rate = read_mono_window(path,start,end-start)
                        waves.append(wave)
                    values = np.asarray(self.registry.predict_windows(job['model_id'],waves,rate),dtype=float)
                    if values.shape!=(len(waves),) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
                        raise ValueError('模型返回了不完整或非法得分')
                    probabilities.extend(values.tolist())
                    self.repo.update_job(job_id,processed_windows=len(probabilities),progress=len(probabilities)/len(windows),
                                         message=f'已分析{len(probabilities)}/{len(windows)}个窗口')
            decoded = decode_events([v[0] for v in windows],[v[1] for v in windows],probabilities,record['audio']['duration_seconds'])
            events = []
            for start,end in decoded['events']:
                scores = [p for left,right,p in zip(decoded['bin_starts'],decoded['bin_ends'],decoded['bin_probabilities'],strict=True)
                          if start<=left and right<=end]
                events.append({'start_seconds':start,'end_seconds':end,'label':'reference_respiratory_event',
                               'score':float(np.mean(scores))})
            with self._lock:
                if flag.is_set():
                    self._cancelled(job_id); return
                self.repo.complete_analysis(job_id,events)
        except Exception as error:
            with self._lock:
                self.repo.update_job(job_id,status='failed',message='分析失败',error=str(error) if isinstance(error,ValueError) else '本地分析执行失败，请检查文件与模型状态',finished_at=utc_now())
        finally:
            with self._lock:
                self._flags.pop(job_id,None)
                self._futures.pop(job_id,None)

    def _cancelled(self,job_id):
        self.repo.update_job(job_id,status='cancelled',message='分析已停止，未发布部分结果',finished_at=utc_now())

    def cancel(self,job_id):
        with self._lock:
            job = self.repo.get_job(job_id)
            if job['status'] in ('completed','cancelled','failed'):
                return job
            flag = self._flags.get(job_id)
            if flag is None:
                raise RuntimeError('任务执行器已不在，请重新打开记录后分析')
            flag.set()
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                self._cancelled(job_id)
                self._flags.pop(job_id,None); self._futures.pop(job_id,None)
            else:
                self.repo.update_job(job_id,message='正在安全停止')
            return self.repo.get_job(job_id)

    def close(self):
        with self._lock:
            for flag in self._flags.values(): flag.set()
        self.executor.shutdown(wait=True,cancel_futures=False)
