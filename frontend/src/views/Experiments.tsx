import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Experiment } from '../types';
const metricNames: Record<string, string> = { macro_f1: 'Macro F1', f1_macro: 'Macro F1', balanced_accuracy: '平衡准确率', average_precision: 'AP', pr_auc_average_precision: 'AP', event_f1: '事件 F1', false_positives_per_hour: '误报 / 小时', fp_per_hour: '误报 / 小时', macro_average_f1: 'Macro F1', macro_ap: 'Macro AP' };
function metrics(value: unknown, path = 'report'): {
    path: string;
    name: string;
    value: number;
}[] { if (!value || typeof value !== 'object')
    return []; return Object.entries(value).flatMap(([k, v]) => typeof v === 'number' && metricNames[k] ? [{ path, name: metricNames[k], value: v }] : typeof v === 'object' && v !== null ? metrics(v, `${path} / ${k}`) : []); }
function metadata(value: unknown, path = ''): {
    name: string;
    value: unknown;
}[] { if (!value || typeof value !== 'object' || Array.isArray(value))
    return []; return Object.entries(value).flatMap(([k, v]) => /seed|cohort|subject|split|dataset|version|parameter|hyperparameter|config|protocol/i.test(k) ? [{ name: `${path}${k}`, value: v }] : typeof v === 'object' && v !== null ? metadata(v, `${path}${k} / `) : []); }
function safeFigure(url: string) { return /^\/api\/experiments\/[^/]+\/figures\/[^/?#]+$/.test(url) ? url : null; }
export function Experiments() {
    const [list, setList] = useState<Experiment[]>([]);
    const [selected, setSelected] = useState('');
    const [detail, setDetail] = useState<Experiment | null>(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);
    const [retry, setRetry] = useState(0);
    useEffect(() => { const c = new AbortController(); setLoading(true); api<{
        experiments: Experiment[];
    }>('/experiments', { signal: c.signal }).then(r => { setList(r.experiments); setSelected(s => s || r.experiments[0]?.id || ''); setError(''); }).catch(e => { if (e.name !== 'AbortError')
        setError(e.message); }).finally(() => { if (!c.signal.aborted)
        setLoading(false); }); return () => c.abort(); }, [retry]);
    useEffect(() => { if (!selected)
        return; const c = new AbortController(); setDetail(null); setLoading(true); api<Experiment>(`/experiments/${encodeURIComponent(selected)}`, { signal: c.signal }).then(r => { setDetail(r); setError(''); }).catch(e => { if (e.name !== 'AbortError')
        setError(e.message); }).finally(() => { if (!c.signal.aborted)
        setLoading(false); }); return () => c.abort(); }, [selected, retry]);
    const rows = metrics(detail?.report);
    const meta = metadata(detail?.report);
    return <><div className="page-heading"><div><div className="eyebrow">RESEARCH / ADVANCED</div><h1>让比较回到证据</h1><p>查看已登记实验的真实报告、分组与研究图表。</p></div><span className="badge">高级功能</span></div><div className="notice">实验结果来自公开数据上的既有运行，不代表个人录音准确率，也不构成医学验证。</div>{error && <div className="notice error" role="alert">{error}<button onClick={() => setRetry(x => x + 1)}>重试</button></div>}<div className="experiment-layout"><aside className="card experiment-list"><h2>已登记实验</h2>{!list.length && !loading && <p className="muted">尚无已登记的研究报告。</p>}{list.map(e => <button className={`experiment-item ${selected === e.id ? 'selected' : ''}`} key={e.id} onClick={() => setSelected(e.id)}><small>{e.category || '研究实验'}</small><strong>{e.title}</strong><span>{e.description || e.dataset || e.id}</span></button>)}</aside><div>{loading ? <div role="status" className="card">正在读取真实实验报告…</div> : detail ? <><section className="card"><div className="eyebrow">REGISTERED REPORT</div><h2>{detail.title}</h2>{rows.length ? <div className="table-scroll"><table><thead><tr><th>模型 / 条件 / 划分</th><th>指标</th><th>实际数值</th></tr></thead><tbody>{rows.map((r, i) => <tr key={i}><td>{r.path.replace(/^report \/ /, '')}</td><td>{r.name}</td><td className="numeric">{r.value.toFixed(4)}</td></tr>)}</tbody></table></div> : <p className="muted">报告未包含可识别的标准指标。请展开原始报告查看实际结果。</p>}<p className="footnote">保留报告中的模型、条件与 split 路径；跨数据集、分组和协议的数值不能直接排名。</p></section>{detail.figures?.filter(f => safeFigure(f.url)).map(f => <figure className="card research-figure" key={f.name}><img loading="lazy" src={safeFigure(f.url)!} alt={f.name} onError={e => { e.currentTarget.hidden = true; e.currentTarget.nextElementSibling?.removeAttribute('hidden'); }}/><p hidden className="error-text">图表加载失败，请检查本地报告文件。</p><figcaption>{f.name}</figcaption></figure>)}<section className="card"><details><summary>数据分组、种子与实验参数</summary>{meta.length ? meta.map((m, i) => <div key={i}><h3>{m.name}</h3><pre>{JSON.stringify(m.value, null, 2)}</pre></div>) : <p className="muted">请查看原始报告中的元信息。</p>}</details><details><summary>完整原始 JSON 报告</summary><pre>{JSON.stringify(detail.report, null, 2)}</pre></details></section></> : <div className="card empty"><h2>选择一个实验</h2><p>这里只呈现已完成并登记的本地研究报告。</p></div>}</div></div></>;
}
