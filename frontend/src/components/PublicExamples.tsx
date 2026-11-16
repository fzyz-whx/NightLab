import { useEffect, useState } from 'react';
import { api, post } from '../api';
import type { NightRecord } from '../types';

interface Example {
    id: string;
    name: string;
    available: boolean;
    reason: string | null;
}

export function PublicExamples({ onImported }: { onImported: (id: string) => Promise<void> }) {
    const [examples, setExamples] = useState<Example[]>([]);
    const [error, setError] = useState('');
    const [busy, setBusy] = useState(false);
    const [loading, setLoading] = useState(true);
    const [retry, setRetry] = useState(0);
    useEffect(() => {
        const controller = new AbortController();
        setLoading(true);
        api<{ examples: Example[] }>('/public-examples', { signal: controller.signal })
            .then(value => { setExamples(value.examples); setError(''); })
            .catch(e => { if (e.name !== 'AbortError') setError(e.message); })
            .finally(() => { if (!controller.signal.aborted) setLoading(false); });
        return () => controller.abort();
    }, [retry]);
    const load = async (id: string) => {
        setBusy(true); setError('');
        try {
            const record = await post<NightRecord>(`/public-examples/${encodeURIComponent(id)}/import`);
            await onImported(record.id);
        } catch (e) { setError((e as Error).message); }
        finally { setBusy(false); }
    };
    return <section className="card public-examples">
        <div className="section-heading"><div><span className="eyebrow">PUBLIC DATA / RESEARCH</span><h2>从公开研究样本开始</h2></div><span className="badge reference">公开数据</span></div>
        <p className="muted">载入 PSG-Audio 验证集固定的前两分钟，比较 PSG 呼吸参考与模型预测。无需个人录音。SleepAccel 的独立睡眠阶段实验在「实验比较」查看。</p>
        {loading && <p role="status">正在检查本地公开样本…</p>}
        {error && <div className="notice error" role="alert">{error}<button disabled={busy || loading} onClick={() => setRetry(n => n + 1)}>重试读取</button></div>}
        <div className="actions">{examples.map(example => <button key={example.id} disabled={busy || !example.available} title={example.reason || '载入前核验官方摘要和准备收据'} onClick={() => void load(example.id)}>{example.name}</button>)}</div>
        {busy && <p role="status">正在校验本地原始文件并载入两分钟片段，请稍候…</p>}
        {!loading && examples.length > 0 && !examples.some(e => e.available) && <p className="footnote">尚未准备公开原始数据；请按安装文档完成获取与校验。当前可直接查看已登记实验，或体验下方合成示例。</p>}
        <p className="footnote">CC BY 4.0 · 仅使用已批准的本地文件 · 不自动下载。参考来自 PSG RML，不是逐声标注或诊断；此处不重新计算正式测试指标。</p>
    </section>;
}
