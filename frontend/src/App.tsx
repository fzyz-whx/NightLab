import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import type { NightRecord } from './types';
import { Records } from './views/Records';
import { RecordViewer } from './views/RecordViewer';
import { Experiments } from './views/Experiments';
export function App() {
    const [page, setPage] = useState('records');
    const [records, setRecords] = useState<NightRecord[]>([]);
    const [selected, setSelected] = useState<string | null>(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);
    const refresh = useCallback(async () => { setLoading(true); try {
        const res = await api<{
            records: NightRecord[];
        }>('/records');
        setRecords(res.records);
        setError('');
    }
    catch (e) {
        setError((e as Error).message);
    }
    finally {
        setLoading(false);
    } }, []);
    useEffect(() => { void refresh(); }, [refresh]);
    const select = (id: string) => { setSelected(id); };
    return <div className="shell"><aside className="sidebar"><a className="brand" href="#" onClick={e => e.preventDefault()}><span className="brand-mark">N</span><span>NightLab<small>夜间研究工作台</small></span></a><div className="nav-label">工作空间</div><nav aria-label="主导航">{[['records', '夜间记录', '01'], ['events', '事件检查', '02'], ['experiments', '实验比较', '03']].map(([id, label, n]) => <button key={id} className={page === id ? 'nav active' : 'nav'} onClick={() => setPage(id)}><span>{n}</span>{label}{id === 'experiments' && <small>高级</small>}</button>)}</nav><div className="local-note"><span className="status-dot"/>本地优先<small>文件与分析保留在本机<br />探索性研究 · 非诊断工具</small></div></aside><main><header className="topbar"><span>个人研究 / {page === 'experiments' ? '公开实验' : '音频与可穿戴记录'}</span><span className="pill">LOCAL WORKSPACE</span></header><div className="content">{page === 'experiments' ? <Experiments /> : <><div className="page-heading"><div><div className="eyebrow">{page === 'events' ? 'EVENT REVIEW' : 'NIGHT RECORDS'}</div><h1>{page === 'events' ? '逐段检查，理解每个标记' : '从一段夜间记录开始'}</h1><p>{page === 'events' ? '将声音、频谱与标记放在同一条时间线上。' : '导入音频，查看夜间声音与已有的可穿戴时序。'}</p></div></div>{error && <div role="alert" className="notice error">{error}<button onClick={() => void refresh()}>重新连接</button></div>}{page === 'records' && <Records records={records} loading={loading} selected={selected} onSelect={select} onImported={async (id) => { await refresh(); select(id); }}/>}{page === 'events' && <div className="record-picker"><label htmlFor="record-pick">当前记录</label><select id="record-pick" value={selected || ''} onChange={e => select(e.target.value)}><option value="">选择一条夜间记录</option>{records.map(r => <option key={r.id} value={r.id}>{r.name}{r.is_demo ? ' · 合成示例' : ''}</option>)}</select></div>}{selected ? <RecordViewer key={selected} id={selected} review={page === 'events'} onDeleted={async () => { setSelected(null); await refresh(); }}/> : page === 'events' && <div className="empty card"><div className="empty-icon">⌁</div><h2>选择一条记录来检查</h2><p>可以先到「夜间记录」导入 WAV，或生成明确标注的合成示例。</p><button onClick={() => setPage('records')}>前往夜间记录</button></div>}</>}</div><footer>NightLab / 数据有来源，结论有边界。</footer></main></div>;
}
