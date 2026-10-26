import { useRef, useState } from 'react';
import { post, uploadAudio } from '../api';
import { time, type NightRecord } from '../types';
interface Props {
    records: NightRecord[];
    loading: boolean;
    selected: string | null;
    onSelect: (id: string) => void;
    onImported: (id: string) => Promise<void>;
}
export function Records({ records, loading, selected, onSelect, onImported }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [consent, setConsent] = useState(false);
    const startInput = useRef<HTMLInputElement>(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [progress, setProgress] = useState<[
        number,
        number
    ] | null>(null);
    const cancel = useRef<(() => void) | null>(null);
    const fileInput = useRef<HTMLInputElement>(null);
    const run = async (demo = false) => { setBusy(true); setError(''); try {
        let r: NightRecord;
        if (demo)
            r = await post('/demo');
        else {
            if (!file || !consent)
                throw new Error('请选择 WAV 并确认录音使用权。');
            const task = uploadAudio(file, startInput.current?.value || '', (a, b) => setProgress([a, b]));
            cancel.current = task.cancel;
            r = await task.promise;
        }
        await onImported(r.id);
        setFile(null);
        if (fileInput.current) fileInput.current.value = '';
        setProgress(null);
    }
    catch (e) {
        setError((e as Error).message);
    }
    finally {
        setBusy(false);
        setProgress(null);
        cancel.current = null;
    } };
    return <section className="records-grid"><div className="card import-card"><div className="section-heading"><div><span className="eyebrow">IMPORT AUDIO</span><h2>添加夜间记录</h2></div><span className="pill">WAV</span></div><label className="file-drop"><span className="upload-symbol">↑</span><strong>{file?.name || '选择本机 WAV 文件'}</strong><small>PCM / float · 8–96 kHz · 最长 24 小时 · 最大 4 GiB</small><input ref={fileInput} aria-label="选择 WAV 音频" type="file" accept=".wav,.wave,audio/wav" disabled={busy} onChange={e => setFile(e.target.files?.[0] || null)}/></label><label className="field">录音开始时间 <span>可选，用于对齐绝对时间附件</span><input ref={startInput} type="datetime-local" step="1" disabled={busy}/></label><label className="checkbox"><input type="checkbox" checked={consent} disabled={busy} onChange={e => setConsent(e.target.checked)}/><span>我确认对录音及所有可能被录到的人具有使用权。</span></label>{progress && <div className="upload-progress"><progress value={progress[0]} max={progress[1]}/><small>已上传 {(progress[0] / 1048576).toFixed(1)} / {(progress[1] / 1048576).toFixed(1)} MiB{progress[0] === progress[1] ? ' · 正在校验 WAV…' : ''}</small></div>}{error && <p role="alert" className="error-text">{error}</p>}<div className="actions"><button className="primary" disabled={busy || !file || !consent} onClick={() => void run()}>{busy ? '正在处理…' : '导入记录'}</button>{busy && cancel.current && <button onClick={() => cancel.current?.()}>取消上传</button>}<button disabled={busy} onClick={() => void run(true)}>生成合成示例</button></div><p className="footnote">示例由本地服务生成，所有片段始终标注为 Demo。不会自动播放。</p></div><div className="card record-list"><div className="section-heading"><h2>记录库</h2><span className="muted">{records.length} 条记录</span></div>{loading ? <p className="muted" role="status">正在读取记录…</p> : records.length === 0 ? <div className="empty compact"><span className="empty-icon">⌁</span><h3>这里还没有记录</h3><p>从左侧导入音频，或先体验合成示例。</p></div> : <div className="record-scroll">{records.map(r => <button className={`record-row ${selected === r.id ? 'selected' : ''}`} key={r.id} onClick={() => onSelect(r.id)}><span className="record-glyph">≋</span><span className="record-copy"><strong>{r.name}</strong><small>{new Date(r.created_at).toLocaleString('zh-CN')} · {time(r.audio.duration_seconds)}</small></span>{r.is_demo && <span className="badge demo">合成 Demo</span>}<span aria-hidden="true">›</span></button>)}</div>}</div></section>;
}
