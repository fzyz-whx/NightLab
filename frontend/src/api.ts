export function errorText(value: unknown): string {
    if (typeof value === 'string')
        return value;
    if (Array.isArray(value))
        return value.map(v => typeof v === 'object' && v !== null && 'msg' in v ? `${'loc' in v && Array.isArray(v.loc) ? v.loc.join(' / ') + ': ' : ''}${v.msg}` : errorText(v)).join('；');
    if (value && typeof value === 'object' && 'detail' in value)
        return errorText(value.detail);
    return '请求未完成，请检查输入或本地服务状态后重试。';
}
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
    try {
        const res = await fetch(`/api${path}`, { ...init, headers: { ...(init.method && init.method !== 'GET' ? { 'X-Nightlab-Client': '1' } : {}), ...init.headers } });
        if (!res.ok) {
            let body: unknown;
            try {
                body = await res.json();
            }
            catch {
                body = `请求失败（${res.status}），请检查本地服务。`;
            }
            throw new Error(errorText(body));
        }
        return await res.json() as T;
    }
    catch (e) {
        if (e instanceof Error && e.name === 'AbortError')
            throw e;
        throw new Error(e instanceof TypeError ? '无法连接本地服务，请确认 NightLab 后端正在运行。' : e instanceof Error ? e.message : errorText(e));
    }
}
export const post = <T>(path: string, body: unknown = {}) => api<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
export function uploadAudio(file: File, start: string, onProgress: (loaded: number, total: number) => void) { const xhr = new XMLHttpRequest(); const promise = new Promise<import('./types').NightRecord>((resolve, reject) => { xhr.open('POST', '/api/records/import'); xhr.setRequestHeader('X-Nightlab-Client', '1'); xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name)); xhr.setRequestHeader('X-Consent', 'true'); xhr.setRequestHeader('Content-Type', 'audio/wav'); if (start)
    xhr.setRequestHeader('X-Start-Time', new Date(start).toISOString()); xhr.upload.onprogress = e => onProgress(e.loaded, e.lengthComputable ? e.total : file.size); xhr.onload = () => { let body: unknown; try {
    body = JSON.parse(xhr.responseText);
}
catch {
    reject(new Error('服务返回了无法读取的响应，请检查本地服务。'));
    return;
} if (xhr.status >= 200 && xhr.status < 300)
    resolve(body as import('./types').NightRecord);
else
    reject(new Error(errorText(body))); }; xhr.onerror = () => reject(new Error('上传连接失败，请检查本地服务。')); xhr.onabort = () => reject(new Error('已取消上传。')); xhr.send(file); }); return { promise, cancel: () => xhr.abort() }; }
