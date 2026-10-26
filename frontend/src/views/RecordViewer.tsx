import { useCallback, useEffect, useRef, useState } from 'react';
import { api, post } from '../api';
import { activeJob, time, type Job, type Model, type RecordDetail, type Waveform } from '../types';
import { Wave, WearablePanel } from '../components/Signals';
export function RecordViewer({ id, review, onDeleted }: {
    id: string;
    review: boolean;
    onDeleted: () => Promise<void>;
}) {
    const [detail, setDetail] = useState<RecordDetail | null>(null);
    const [wave, setWave] = useState<Waveform | null>(null);
    const [models, setModels] = useState<Model[]>([]);
    const [model, setModel] = useState('');
    const [error, setError] = useState('');
    const [busy, setBusy] = useState(false);
    const [job, setJob] = useState<Job | null>(null);
    const [range, setRange] = useState<[
        number,
        number
    ]>([0, 10]);
    const [current, setCurrent] = useState(0);
    const [label, setLabel] = useState('');
    const [attachment, setAttachment] = useState<File | null>(null);
    const [unit, setUnit] = useState('relative_seconds');
    const [deleting, setDeleting] = useState(false);
    const [spectrumError, setSpectrumError] = useState(false);
    const audio = useRef<HTMLAudioElement>(null);
    const clipEnd = useRef<number | null>(null);
    const playSelection = async () => { if (!audio.current) return; audio.current.currentTime = range[0]; clipEnd.current = range[1]; try { await audio.current.play(); } catch { clipEnd.current = null; setError("播放未能开始，请检查音频文件或浏览器权限。"); } };
    const load = useCallback(async () => { const value = await api<RecordDetail>(`/records/${id}`); setDetail(value); setJob(value.latest_job); return value; }, [id]);
    useEffect(() => { const control = new AbortController(); setError(''); Promise.all([api<RecordDetail>(`/records/${id}`, { signal: control.signal }), api<Waveform>(`/records/${id}/waveform?points=1600`, { signal: control.signal }), api<{
            models: Model[];
        }>('/models', { signal: control.signal })]).then(([d, w, m]) => { setDetail(d); setJob(d.latest_job); setWave(w); setModels(m.models); setModel(m.models.find(x => x.available)?.id || ''); setRange([0, Math.min(10, d.record.audio.duration_seconds)]); }).catch(e => { if (e.name !== 'AbortError')
        setError(e.message); }); return () => control.abort(); }, [id]);
    useEffect(() => { if (!job || !activeJob(job))
        return; let stopped = false; let timer: ReturnType<typeof setTimeout>; const poll = async () => { try {
        const value = await api<Job>(`/jobs/${job.id}`);
        if (stopped)
            return;
        setJob(value);
        if (activeJob(value))
            timer = setTimeout(() => void poll(), 1000);
        else {
            await load();
        }
    }
    catch (e) {
        if (!stopped) {
            setError((e as Error).message);
            timer = setTimeout(() => void poll(), 3000);
        }
    } }; timer = setTimeout(() => void poll(), 800); return () => { stopped = true; clearTimeout(timer); }; }, [job?.id, job?.status, load]);
    const perform = async (fn: () => Promise<void>) => { setError(''); setBusy(true); try {
        await fn();
    }
    catch (e) {
        setError((e as Error).message);
    }
    finally {
        setBusy(false);
    } };
    const seek = (t: number) => { clipEnd.current = null; setCurrent(t); if (audio.current)
        audio.current.currentTime = t; const duration = detail?.record.audio.duration_seconds || 10; setRange([Math.min(t, Math.max(0, duration - 1)), Math.min(duration, t + 10)]); setSpectrumError(false); };
    if (!detail)
        return <section className="card">{error ? <div className="notice error" role="alert">{error}<button onClick={() => void perform(async () => { await load(); })}>重试</button></div> : <p role="status">正在加载记录与波形…</p>}</section>;
    const r = detail.record;
    const duration = r.audio.duration_seconds;
    const selectedModel = models.find(m => m.id === model);
    return <div className="viewer"><section className="card audio-card"><div className="section-heading"><div><span className="eyebrow">{review ? 'EVENT INSPECTION' : 'RECORD DETAIL'}</span><h2>{r.name}</h2><p className="muted">{time(duration)} · {(r.audio.sample_rate / 1000).toFixed(1)} kHz · {r.audio.channels === 1 ? '单声道' : '双声道'}{r.start_time ? ` · 开始 ${new Date(r.start_time).toLocaleString('zh-CN')}` : ' · 未设置绝对开始时间'}</p></div><div className="actions">{r.is_demo && <span className="badge demo">合成 Demo · 非真实录音</span>}<button className="danger subtle" onClick={() => setDeleting(true)}>删除记录</button></div></div>{error && <div role="alert" className="notice error">{error}<button aria-label="关闭错误" onClick={() => setError('')}>×</button></div>}<Wave wave={wave} duration={duration} current={current} range={range} onSeek={seek}/><div className="player"><audio key={id} ref={audio} controls preload="metadata" src={`/api/records/${id}/audio`} onTimeUpdate={e => { const player = e.currentTarget; if (clipEnd.current !== null && player.currentTime >= clipEnd.current) { player.pause(); player.currentTime = clipEnd.current; clipEnd.current = null; } setCurrent(player.currentTime); }} onError={() => setError('音频未能加载，请检查本地文件与服务。')}/><span className="time-readout">{time(current)} / {time(duration)}</span><button disabled={range[0] < 0 || range[1] <= range[0] || range[1] > duration} onClick={() => void playSelection()}>播放选段</button></div><p className="footnote">点击波形或片段可定位；「播放选段」会在选段结束后暂停。播放由你手动开始。</p></section><div className="inspection-grid"><section className="card"><div className="section-heading"><h2>声音片段</h2><a className="button-link" href={`/api/records/${id}/events.csv`} download>导出 CSV ↓</a></div><div className="legend"><span><i className="dot model"/>模型预测</span><span><i className="dot user"/>用户标记</span><span><i className="dot demo"/>合成示例</span></div>{detail.events.length === 0 ? <div className="missing"><strong>尚无片段标记</strong><p>可运行本地模型分析，或手动添加观察。</p></div> : <div className="event-list">{detail.events.map(e => <div className="event-row" key={e.id}><button className="event-target" onClick={() => { seek(e.start_seconds); setRange([e.start_seconds, Math.min(e.end_seconds, e.start_seconds + 60)]); setSpectrumError(false); }}><span className={`badge ${e.source}`}>{e.source === 'model' ? '模型预测' : e.source === 'user' ? '用户标记' : '合成 Demo'}</span><strong>{e.source === 'model' && e.label === 'reference_respiratory_event' ? '呼吸相关事件候选' : e.label}</strong><small>{time(e.start_seconds)} – {time(e.end_seconds)}{e.source === 'model' ? ` · 得分 ${e.score === null ? '未提供' : e.score.toFixed(3)} · ${e.model_version || models.find(m => m.id === e.model_id)?.version || e.model_id || '版本未提供'}` : ''}</small></button>{e.source !== 'model' && <button className="subtle" aria-label={`删除 ${e.label} 标记`} disabled={busy} onClick={() => void perform(async () => { await api(`/records/${id}/annotations/${e.id}`, { method: 'DELETE' }); await load(); })}>删除</button>}</div>)}</div>}<form className="annotation-form" onSubmit={e => { e.preventDefault(); void perform(async () => { await post(`/records/${id}/annotations`, { start_seconds: range[0], end_seconds: range[1], label }); setLabel(''); await load(); }); }}><h3>添加用户标记</h3><div className="range-fields"><label>开始（秒）<input aria-label="标记开始秒数" type="number" min="0" max={duration} step="0.1" value={range[0]} onChange={e => { setRange([Number(e.target.value), range[1]]); setSpectrumError(false); }}/></label><label>结束（秒）<input aria-label="标记结束秒数" type="number" min="0" max={duration} step="0.1" value={range[1]} onChange={e => { setRange([range[0], Number(e.target.value)]); setSpectrumError(false); }}/></label></div><input aria-label="用户标记名称" placeholder="例如：环境声音、待复核片段" required maxLength={120} value={label} onChange={e => setLabel(e.target.value)}/><button disabled={busy || !label.trim() || range[1] <= range[0] || range[1] > duration || range[0] < 0}>保存用户标记</button></form></section><section className="card"><div className="section-heading"><div><h2>局部声谱图</h2><p className="muted">{time(range[0])} 起 · 最多 60 秒</p></div><span className="pill">相对强度</span></div>{range[1] > range[0] && range[0] >= 0 && range[0] < duration ? spectrumError ? <div className="missing">声谱图加载失败。<button onClick={() => setSpectrumError(false)}>重试</button></div> : <img className="spectrogram" alt="当前选段的相对频谱图，未进行声压级标定" src={`/api/records/${id}/spectrogram?start=${range[0]}&duration=${Math.min(60, range[1] - range[0], duration - range[0])}`} onError={() => setSpectrumError(true)}/> : <p className="muted">请设置音频范围内有效的起止时间。</p>}<p className="footnote">用于检查声音结构；颜色表示相对频谱强度，未经 SPL 标定。</p><div className="analysis-box"><h3>本地模型分析</h3><label className="field">选择可用模型<select value={model} disabled={busy || activeJob(job)} onChange={e => setModel(e.target.value)}><option value="">{models.some(m => m.available) ? '选择模型' : '暂无可用模型'}</option>{models.map(m => <option disabled={!m.available} key={m.id} value={m.id}>{m.name} · {m.version}{m.available ? '' : '（不可用）'}</option>)}</select></label>{selectedModel && <p className="footnote">{selectedModel.scope} · 默认 CPU</p>}{models.filter(m => !m.available).map(m => <p className="footnote" key={m.id}>{m.name}：{m.reason || '本机模型不可用'}</p>)}<button className="primary" disabled={busy || !selectedModel?.available || activeJob(job)} onClick={() => void perform(async () => setJob(await post<Job>(`/records/${id}/analyze`, { model_id: model })))}>开始分析</button>{job && <div className="job" role="status"><div className="section-heading"><strong>{{ queued: '等待执行', running: '正在分析', completed: '分析完成', cancelled: '已取消', failed: '分析失败' }[job.status]}</strong>{activeJob(job) && <button disabled={busy} onClick={() => void perform(async () => setJob(await post<Job>(`/jobs/${job.id}/cancel`)))}>取消任务</button>}</div><progress max={Math.max(1, job.total_windows)} value={job.processed_windows}/><small>{job.processed_windows} / {job.total_windows} 个窗口 · {job.message}</small>{job.error && <p role="alert" className="error-text">{job.error}</p>}</div>}<p className="footnote">输出为模型预测与模型得分。个人记录没有参考真值，不计算准确率或健康分数。</p></div></section></div><WearablePanel wearable={detail.wearable} duration={duration}/><section className="card attachment"><div><h2>添加可穿戴附件</h2><p className="muted">CSV / JSON · 最大 20 MiB。使用明确单位，绝对时间要求录音已有开始时间。</p><a href="/api/templates/wearable.csv" download>下载 CSV 模板</a><span> · </span><a href="/api/templates/wearable.json" download>下载 JSON 模板</a></div><form onSubmit={e => { e.preventDefault(); void perform(async () => { if (!attachment)
        return; await api(`/records/${id}/wearable?kind=combined&time_unit=${unit}`, { method: 'POST', headers: { 'Content-Type': attachment.name.toLowerCase().endsWith('.json') ? 'application/json' : 'text/csv' }, body: attachment }); await load(); }); }}><input aria-label="选择可穿戴附件" type="file" accept=".csv,.json" onChange={e => setAttachment(e.target.files?.[0] || null)}/><select aria-label="附件时间单位" value={unit} onChange={e => setUnit(e.target.value)}><option value="relative_seconds">相对录音开始 · 秒</option><option value="epoch_seconds">Unix 时间 · 秒</option><option value="epoch_milliseconds">Unix 时间 · 毫秒</option><option value="iso8601">ISO 8601 · 带时区</option></select><button disabled={!attachment || busy}>导入附件</button></form></section>{deleting && <div className="modal-backdrop"><section role="dialog" aria-modal="true" aria-labelledby="delete-title" className="modal"><h2 id="delete-title">删除这条记录？</h2><p>将删除「{r.name}」的原始音频、附件、衍生结果与标记。此操作无法撤销。</p>{activeJob(job) && <p className="notice">此记录有活动任务。请先取消任务，等待状态变为已取消或结束后再删除。</p>}<div className="actions"><button disabled={busy} onClick={() => setDeleting(false)}>保留记录</button>{activeJob(job) ? <button disabled={busy} onClick={() => void perform(async () => setJob(await post<Job>(`/jobs/${job!.id}/cancel`)))}>请求取消任务</button> : <button className="danger" disabled={busy} onClick={() => void perform(async () => { const result = await api<{
        deleted: boolean;
    }>(`/records/${id}`, { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirm: true }) }); if (!result.deleted)
        throw new Error('服务尚未确认删除，请刷新后重试。'); audio.current?.pause(); if (audio.current) {
        audio.current.removeAttribute('src');
        audio.current.load();
    } await onDeleted(); })}>确认删除</button>}</div></section></div>}</div>;
}
