"""Private local imports, real media/model jobs, and read-only experiment views."""
import asyncio
from contextlib import asynccontextmanager
import csv
from functools import partial
import io
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
from urllib.parse import unquote
from uuid import uuid4

from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,ConfigDict,Field,FiniteFloat,StrictBool
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import ClientDisconnect

from nightlab.services.records import RecordService
from nightlab.services import media,wearable_import
from nightlab.services.model_registry import ModelRegistry
from nightlab.services.jobs import JobManager
from nightlab.services.experiments import ExperimentService


class StrictBody(BaseModel):
    model_config = ConfigDict(extra='forbid')


class AnalyzeBody(StrictBody):
    model_id: str = 'audio_rf_pilot_v1'


class DeleteBody(StrictBody):
    confirm: StrictBool = False


class AnnotationBody(StrictBody):
    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(gt=0)
    label: str = Field(min_length=1,max_length=120)


def create_app(data_dir=None,project_root=None,num_threads=2,registry=None):
    root = Path(project_root or Path(__file__).resolve().parents[4]).resolve()
    directory = Path(data_dir or os.environ.get('NIGHTLAB_DATA_DIR',root/'private/app')).resolve()
    records = RecordService(directory)
    models = registry or ModelRegistry(root,num_threads)
    jobs = JobManager(records,models,num_threads)
    experiments = ExperimentService(root)
    imports = set()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            for flag in list(imports): flag.set()
            jobs.close()

    app = FastAPI(title='NightLab local research workbench',version='0.1.0.dev0',lifespan=lifespan,
                  docs_url=None,redoc_url=None,openapi_url='/api/openapi.json',
                  telemetry={'tracing':False,'metrics':False,'logs':False,'operation_spans':False,'auto_configure':False})
    app.state.records,app.state.jobs,app.state.models = records,jobs,models
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','testserver','[::1]'])
    app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5178','http://localhost:5178'],
        allow_methods=['GET','POST','DELETE'],allow_headers=['Content-Type','X-Nightlab-Client','X-Filename','X-Start-Time','X-Consent'])

    @app.middleware('http')
    async def local_requests(request,call_next):
        if request.method not in ('GET','HEAD','OPTIONS') and request.headers.get('x-nightlab-client')!='1':
            return JSONResponse({'detail':'仅接受本地工作台的明确操作'},status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    @app.exception_handler(ValueError)
    async def invalid(_request,error):
        return JSONResponse({'detail':str(error)},status_code=400)

    @app.exception_handler(KeyError)
    async def missing(_request,_error):
        return JSONResponse({'detail':'记录或资源不存在'},status_code=404)

    @app.exception_handler(FileNotFoundError)
    async def missing_file(_request,_error):
        return JSONResponse({'detail':'本地文件或模型尚未准备，请检查记录状态'},status_code=404)

    @app.exception_handler(RuntimeError)
    async def conflict(_request,error):
        return JSONResponse({'detail':str(error)},status_code=409)

    @app.exception_handler(OSError)
    async def io_error(_request,_error):
        return JSONResponse({'detail':'本地文件读写失败，请确认磁盘空间或关闭占用文件后重试'},status_code=409)

    @app.get('/api/health')
    def health():
        return {'status':'ok','app':'NightLab','version':'0.1.0.dev0','privacy':'local'}

    @app.get('/api/records')
    def list_records(): return {'records':records.repo.list_records()}

    @app.post('/api/records/import',status_code=201)
    async def import_audio(request:Request):
        if request.headers.get('x-consent')!='true':
            raise ValueError('请先确认录音使用权与所有可能被录到者的同意')
        name = unquote(request.headers.get('x-filename','夜间录音.wav'),encoding='utf-8',errors='strict').strip()
        if not name or len(name)>120 or any(ord(c)<32 for c in name) or '/' in name or '\\' in name:
            raise ValueError('请使用不含路径与控制字符的文件名（最多120字）')
        length = request.headers.get('content-length')
        if length is not None and (not length.isdigit() or int(length)>media.MAX_BYTES):
            raise HTTPException(413,'音频超过大小限制')
        required = int(length) if length else media.MAX_BYTES
        if shutil.disk_usage(records.staging_dir).free<required+16*1024**2:
            raise HTTPException(507,'本地磁盘剩余空间不足')
        path = records.staging_dir/(str(uuid4())+'.wav')
        flag = threading.Event(); imports.add(flag)
        watcher = None
        try:
            total = 0
            with path.open('xb') as output:
                async for block in request.stream():
                    total += len(block)
                    if total>media.MAX_BYTES:
                        raise HTTPException(413,'音频超过大小限制')
                    output.write(block)
            if not total or (length and total!=int(length)):
                raise ValueError('上传为空或不完整')
            work = asyncio.wrap_future(jobs.executor.submit(partial(records.import_audio,path,name,
                start_time=request.headers.get('x-start-time') or None,consent=True,stop_requested=flag.is_set)))
            async def monitor():
                while not work.done():
                    if await request.is_disconnected(): flag.set(); return
                    await asyncio.sleep(.2)
            watcher = asyncio.create_task(monitor())
            try:
                result = await asyncio.shield(work)
            except asyncio.CancelledError:
                flag.set()
                try: await asyncio.shield(work)
                except (InterruptedError,asyncio.CancelledError): pass
                raise
            if flag.is_set():
                records.delete_record(result['id'],confirmed=True)
                raise ValueError('导入已取消')
            return result
        except ClientDisconnect:
            flag.set()
            raise HTTPException(499,'上传已停止')
        except InterruptedError as error:
            raise HTTPException(499,str(error)) from error
        finally:
            if watcher: watcher.cancel()
            imports.discard(flag)
            if path.exists(): records._safe(path,records.staging_dir).unlink()

    @app.post('/api/demo',status_code=201)
    async def demo():
        def generate():
            path = records.staging_dir/(str(uuid4())+'.wav')
            sample = media.deterministic_demo(path)
            record = records.import_audio(path,'合成夜间示例.wav',consent=True,is_demo=True,
                                          start_time='2026-01-02T23:59:20+08:00')
            for event in sample['events']: records.repo.add_event({**event,'record_id':record['id']})
            records.repo.set_wearable(record['id'],sample['wearable'])
            return record
        return await asyncio.wrap_future(jobs.executor.submit(generate))

    @app.get('/api/records/{record_id}')
    def detail(record_id:str): return records.detail(record_id)

    @app.get('/api/records/{record_id}/audio')
    def audio(record_id:str): return FileResponse(records.audio_path(record_id),media_type='audio/wav',content_disposition_type='inline')

    @app.get('/api/records/{record_id}/waveform')
    def waveform(record_id:str,points:int=1600,start:float=0,end:float|None=None):
        return media.waveform(records.audio_path(record_id),points,start,end)

    @app.get('/api/records/{record_id}/spectrogram')
    def spectrogram(record_id:str,start:float=0,duration:float=10):
        record = records.repo.get_record(record_id)
        if not math.isfinite(start) or not math.isfinite(duration) or start<0 or not 0<duration<=60 or start>=record['audio']['duration_seconds']:
            raise ValueError('请选择录音内不超过60秒的区间')
        content = media.spectrogram_png(records.audio_path(record_id),start,min(duration,record['audio']['duration_seconds']-start))
        return Response(content,media_type='image/png')

    @app.post('/api/records/{record_id}/wearable')
    async def wearable(record_id:str,request:Request,kind:str='combined',time_unit:str='relative_seconds'):
        record = records.repo.get_record(record_id)
        content = bytearray()
        async for block in request.stream():
            content.extend(block)
            if len(content)>20*1024**2: raise HTTPException(413,'时序附件超过20MiB限制')
        payload = wearable_import.parse_wearable(bytes(content),request.headers.get('content-type',''),kind,time_unit,
            record['start_time'],record['audio']['duration_seconds'])
        return records.repo.set_wearable(record_id,payload)

    @app.get('/api/templates/wearable.csv')
    def template_csv(): return Response(wearable_import.template_csv(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=wearable-template.csv'})

    @app.get('/api/templates/wearable.json')
    def template_json(): return Response(wearable_import.template_json(),media_type='application/json',headers={'Content-Disposition':'attachment; filename=wearable-template.json'})

    @app.post('/api/records/{record_id}/annotations',status_code=201)
    def annotation(record_id:str,body:AnnotationBody):
        record = records.repo.get_record(record_id)
        if not body.label.strip() or body.end_seconds<=body.start_seconds or body.end_seconds>record['audio']['duration_seconds']:
            raise ValueError('标记区间须位于录音内，结束晚于开始')
        return records.repo.add_event({**body.model_dump(),'label':body.label.strip(),'record_id':record_id,
                                      'source':'user','score':None,'model_id':None})

    @app.delete('/api/records/{record_id}/annotations/{event_id}')
    def remove_annotation(record_id:str,event_id:str):
        records.repo.delete_event(record_id,event_id); return {'deleted':True}

    @app.get('/api/records/{record_id}/events.csv')
    def csv_events(record_id:str):
        buffer = io.StringIO(newline=''); writer = csv.writer(buffer)
        columns = ('start_seconds','end_seconds','label','source','score','model_id','model_version')
        writer.writerow(columns)
        for event in records.repo.get_events(record_id):
            row = [event.get(name) for name in columns]
            row = [("'"+str(value)) if isinstance(value,str) and value.startswith(('=','+','-','@')) else value for value in row]
            writer.writerow(row)
        return Response('\ufeff'+buffer.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=events.csv'})

    @app.get('/api/models')
    def list_models(): return {'models':models.list_models()}

    @app.post('/api/records/{record_id}/analyze',status_code=202)
    def analyze(record_id:str,body:AnalyzeBody): return jobs.submit(record_id,body.model_id)

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id:str): return records.repo.get_job(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id:str): return jobs.cancel(job_id)

    @app.delete('/api/records/{record_id}')
    def delete_record(record_id:str,body:DeleteBody): return records.delete_record(record_id,body.confirm)

    @app.get('/api/experiments')
    def list_experiments(): return {'experiments':experiments.list()}

    @app.get('/api/experiments/{experiment_id}')
    def experiment(experiment_id:str): return experiments.detail(experiment_id)

    @app.get('/api/experiments/{experiment_id}/figures/{name}')
    def figure(experiment_id:str,name:str): return FileResponse(experiments.figure(experiment_id,name),media_type='image/png')

    dist = root/'frontend/dist'
    if dist.is_dir(): app.mount('/',StaticFiles(directory=dist,html=True),name='workbench')
    return app
