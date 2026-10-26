import type { Wearable, Waveform } from '../types';
import { time } from '../types';
export function Wave({ wave, duration, current, range, onSeek }: {
    wave: Waveform | null;
    duration: number;
    current: number;
    range: [
        number,
        number
    ];
    onSeek: (t: number) => void;
}) { const width = 1000; return <div className="wave"><svg role="slider" aria-label="音频波形，左右键移动五秒" aria-valuemin={0} aria-valuemax={duration} aria-valuenow={current} tabIndex={0} viewBox="0 0 1000 140" preserveAspectRatio="none" onClick={e => { const rect = e.currentTarget.getBoundingClientRect(); onSeek(Math.max(0, Math.min(duration, (e.clientX - rect.left) / rect.width * duration))); }} onKeyDown={e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    onSeek(Math.max(0, Math.min(duration, current + (e.key === 'ArrowRight' ? 5 : -5))));
} }}><line x1="0" x2="1000" y1="70" y2="70" stroke="#dce4e3"/><rect x={range[0] / duration * width} y="0" width={(range[1] - range[0]) / duration * width} height="140" fill="#e1eeea"/>{wave?.points.map((p, i) => <line key={i} x1={p.time_seconds / duration * width} x2={p.time_seconds / duration * width} y1={70 - p.max * 62} y2={70 - p.min * 62} stroke="#387f79" strokeWidth="1.1"/>)}<line x1={current / duration * width} x2={current / duration * width} y1="0" y2="140" stroke="#d18045" strokeWidth="2"/></svg><div className="time-scale"><span>00:00:00</span><span>{time(duration / 2)}</span><span>{time(duration)}</span></div></div>; }
function lineSegments(rows: {
    time_seconds: number;
    value: number | null;
}[], duration: number, min = 30, max = 180) { const deltas = rows.slice(1).map((r, i) => r.time_seconds - rows[i].time_seconds).filter(d => d > 0).sort((a, b) => a - b); const maxGap = (deltas[Math.floor(deltas.length / 2)] || 60) * 2.5; const paths: string[] = []; let d = ''; let previous: number | null = null; for (const r of rows) {
    if (r.value === null) {
        if (d)
            paths.push(d.includes('L') ? d : d + 'l0.01,0');
        d = '';
        previous = null;
        continue;
    }
    if (previous !== null && r.time_seconds - previous > maxGap) {
        if (d)
            paths.push(d);
        d = '';
    }
    const x = r.time_seconds / duration * 1000, y = 110 - (r.value - min) / Math.max(1, max - min) * 90;
    d += `${d ? 'L' : 'M'}${x},${y} `;
    previous = r.time_seconds;
} if (d)
    paths.push(d.includes('L') ? d : d + 'l0.01,0'); return paths; }
export function WearablePanel({ wearable, duration }: {
    wearable: Wearable | null;
    duration: number;
}) {
    const heartValues = wearable?.heart_rate?.flatMap(r => r.value === null ? [] : [r.value]) || [];
    const heartLow = Math.min(30, heartValues.reduce((a,b) => Math.min(a,b), 30));
    const heartHigh = Math.max(180, heartValues.reduce((a,b) => Math.max(a,b), 180));
    return <section className="card"><div className="section-heading"><div><span className="eyebrow">OPTIONAL SIGNALS</span><h2>可穿戴时序</h2></div>{wearable && <span className="badge">导入附件 · 非 PSG 参考</span>}</div>{!wearable ? <div className="missing"><strong>尚未添加可穿戴附件</strong><p>音频仍可正常浏览。添加心率、阶段或加速度 CSV / JSON 后，这里会展示实际可用的观察。</p></div> : <>{wearable.warnings?.map((w, i) => <div className="notice" key={i}>{w}</div>)}{wearable.heart_rate?.length > 0 && <div className="signal"><div className="signal-label">心率 <span>bpm · 空值及较长采样间隔断线</span></div><svg viewBox="0 0 1000 130" role="img" aria-label="导入心率曲线，保留缺口"><text x="0" y="20" className="chart-text">{heartHigh}</text><text x="0" y="113" className="chart-text">{heartLow}</text>{lineSegments(wearable.heart_rate, duration, heartLow, heartHigh).map((d, i) => <path key={i} d={d} fill="none" stroke="#387f79" strokeWidth="2" strokeLinecap="round"/>)}</svg></div>}{wearable.stages?.length > 0 && <div className="signal"><div className="signal-label">阶段 <span>设备 / 用户导入标签</span></div><svg viewBox="0 0 1000 60" role="img" aria-label="睡眠阶段时间轴">{wearable.stages.map((s, i) => <g key={i}><rect x={s.time_seconds / duration * 1000} y="5" width={s.duration_seconds / duration * 1000} height="40" fill={({ W: '#c8b990', N1: '#a9cfca', N2: '#73a7a2', N3: '#345c73', REM: '#a6a5c7', Unknown: '#d9dcdd' } as Record<string, string>)[s.stage] || '#d9dcdd'}/>{s.duration_seconds / duration > .035 && <text x={(s.time_seconds + s.duration_seconds / 2) / duration * 1000} y="30" textAnchor="middle" fontSize="13" fill={s.stage === 'N3' ? 'white' : '#243c4a'}>{s.stage}</text>}</g>)}</svg><small className="muted">W 清醒 · N1 / N2 / N3 · REM · Unknown 未知（灰色）；空白处无观察。</small></div>}{wearable.acceleration?.length > 0 && <div className="signal"><div className="signal-label">加速度 <span>g · 三轴独立保留缺失</span></div><svg viewBox="0 0 1000 130" role="img" aria-label="三轴加速度曲线">{(['x_g', 'y_g', 'z_g'] as const).map((axis, i) => lineSegments(wearable.acceleration.map(r => ({ time_seconds: r.time_seconds, value: r[axis] === null ? null : 90 + (r[axis] as number) * 30 })), duration).map((d, j) => <path key={`${axis}-${j}`} d={d} fill="none" stroke={['#387f79', '#a07749', '#7c7dba'][i]} strokeWidth="1.5"/>))}</svg><small className="muted">X 青绿 · Y 棕 · Z 紫；中心为 0 g，每 18 个纵向图单位为 1 g。</small></div>}{!wearable.heart_rate?.length && !wearable.stages?.length && !wearable.acceleration?.length && <p className="muted">附件仅包含夜间汇总，未生成时序曲线。</p>}{wearable.summary && Object.keys(wearable.summary).length > 0 && <details><summary>附件汇总</summary><pre>{JSON.stringify(wearable.summary, null, 2)}</pre></details>}</>}</section>; }
