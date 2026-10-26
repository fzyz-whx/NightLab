export interface AudioInfo {
    duration_seconds: number;
    sample_rate: number;
    channels: number;
    bytes: number;
    format: string;
    subtype: string;
    frames: number;
    sha256: string;
}
export interface NightRecord {
    id: string;
    name: string;
    created_at: string;
    start_time: string | null;
    is_demo: boolean;
    consent: boolean;
    audio: AudioInfo;
}
export interface Event {
    id: string;
    record_id: string;
    start_seconds: number;
    end_seconds: number;
    label: string;
    source: 'user' | 'demo' | 'model';
    score: number | null;
    model_id: string | null;
    model_version?: string | null;
}
export interface Job {
    id: string;
    record_id: string;
    model_id: string;
    status: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed';
    progress: number;
    processed_windows: number;
    total_windows: number;
    message: string;
    error: string | null;
}
export interface Model {
    id: string;
    name: string;
    kind: string;
    available: boolean;
    reason: string | null;
    version: string;
    scope: string;
}
export interface Wearable {
    source: string;
    heart_rate: {
        time_seconds: number;
        value: number | null;
    }[];
    stages: {
        time_seconds: number;
        duration_seconds: number;
        stage: string;
    }[];
    acceleration: {
        time_seconds: number;
        x_g: number | null;
        y_g: number | null;
        z_g: number | null;
    }[];
    summary: Record<string, unknown>;
    warnings: string[];
}
export interface RecordDetail {
    record: NightRecord;
    events: Event[];
    wearable: Wearable | null;
    latest_job: Job | null;
}
export interface Waveform {
    duration_seconds: number;
    points: {
        time_seconds: number;
        min: number;
        max: number;
    }[];
}
export interface Figure {
    name: string;
    url: string;
}
export interface Experiment {
    id: string;
    title: string;
    description?: string;
    category?: string;
    dataset?: string;
    figures: Figure[];
    report?: unknown;
}
export const activeJob = (job: Job | null) => job?.status === 'queued' || job?.status === 'running';
export const time = (seconds: number) => `${Math.floor(seconds / 3600).toString().padStart(2, '0')}:${Math.floor(seconds % 3600 / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
