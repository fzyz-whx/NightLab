"""Bounded, dependency-free user wearable import; labels are not PSG truth."""

from __future__ import annotations

import csv
from datetime import datetime
import io
import json
import math

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 200_000
TIME_UNITS = {"relative_seconds", "epoch_seconds", "epoch_milliseconds", "iso8601"}
KINDS = {"combined", "heart_rate", "stages", "acceleration", "summary"}
STAGES = {"W", "N1", "N2", "N3", "REM", "Unknown"}
CSV_FIELDS = {"time_seconds", "timestamp", "heart_rate_bpm", "stage", "stage_duration_seconds",
              "accel_x_g", "accel_y_g", "accel_z_g"}


def _number(value, field, nullable=False):
    if value is None or value == "":
        if nullable:
            return None
        raise ValueError(f"{field}不能为空")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field}必须是有限数字")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{field}必须是有限数字") from None
    if not math.isfinite(result):
        raise ValueError(f"{field}必须是有限数字")
    return result


def _iso(value, field):
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"{field}必须是带时区的ISO时间")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError()
        return result
    except (ValueError, OverflowError):
        raise ValueError(f"{field}必须是带时区的ISO时间") from None


def _keys(obj, allowed, field):
    if not isinstance(obj, dict):
        raise ValueError(f"{field}必须是对象")
    unknown = set(obj) - allowed
    if unknown:
        raise ValueError(f"{field}含未知字段：{', '.join(sorted(unknown))}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON字段重复：{key}")
        result[key] = value
    return result


def _summary(value):
    if not isinstance(value, dict) or len(value) > 100:
        raise ValueError("summary必须是最多100项的对象")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 80:
            raise ValueError("summary键名须为1至80字符")
        if isinstance(item, str):
            if len(item) > 1000:
                raise ValueError("summary文本不得超过1000字符")
        elif isinstance(item, bool):
            pass
        elif isinstance(item, (int, float)):
            _number(item, f"summary.{key}")
        else:
            raise ValueError("summary仅接受有限数字、文本或布尔值，请移除个人身份字段")
        result[key] = item
    return result


def parse_wearable(content: bytes, content_type: str, kind: str = "combined",
                   time_unit: str = "relative_seconds", start_time: str | None = None,
                   audio_duration: float | None = None) -> dict:
    """Parse UTF-8 CSV/JSON without guessing units, sorting, clipping or filling gaps."""
    if not isinstance(content, bytes) or len(content) > MAX_BYTES:
        raise ValueError("导入文件必须是字节内容，且不得超过20MiB")
    if kind not in KINDS:
        raise ValueError("kind须为combined/heart_rate/stages/acceleration/summary")
    if time_unit not in TIME_UNITS:
        raise ValueError("不支持的time_unit，请明确时间单位")
    if audio_duration is not None:
        audio_duration = _number(audio_duration, "audio_duration")
        if audio_duration < 0:
            raise ValueError("audio_duration不得为负数")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("文件须使用UTF-8编码（允许BOM）") from None
    output = {"source": "user_import", "time_unit": "relative_seconds", "heart_rate": [],
              "stages": [], "acceleration": [], "summary": {}, "warnings": []}
    synthetic = False
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"application/json", "text/json"}:
        try:
            payload = json.loads(text, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, RecursionError):
            raise ValueError("JSON格式无效，请检查对象和数组") from None
        _keys(payload, {"time_unit", "heart_rate", "stages", "acceleration", "summary", "meta"}, "JSON")
        time_unit = payload.get("time_unit", time_unit)
        if not isinstance(time_unit, str) or time_unit not in TIME_UNITS:
            raise ValueError("JSON time_unit不支持，请明确时间单位")
        if "meta" in payload:
            _keys(payload["meta"], {"synthetic", "description", "acceleration_unit"}, "meta")
            meta = payload["meta"]
            if "synthetic" in meta and not isinstance(meta["synthetic"], bool):
                raise ValueError("meta.synthetic必须是布尔值")
            if "description" in meta and (not isinstance(meta["description"], str) or len(meta["description"]) > 1000):
                raise ValueError("meta.description必须是最多1000字符的文本")
            if meta.get("acceleration_unit", "g") != "g":
                raise ValueError("加速度单位必须明确为g")
            synthetic = meta.get("synthetic", False)
        output["summary"] = _summary(payload.get("summary", {}))
        signals = {}
        count = 0
        for signal in ("heart_rate", "stages", "acceleration"):
            rows = payload.get(signal, [])
            if not isinstance(rows, list):
                raise ValueError(f"{signal}必须是数组")
            count += len(rows)
            if count > MAX_ROWS:
                raise ValueError("导入时序条目不得超过200000条")
            signals[signal] = rows
    elif media_type in {"text/csv", "application/csv"}:
        # Only a leading explicit synthetic marker is accepted, not arbitrary comments.
        if text.startswith("# synthetic: true\n") or text.startswith("# synthetic: true\r\n"):
            synthetic = True
            text = text.split("\n", 1)[1]
        try:
            reader = csv.reader(io.StringIO(text, newline=""), strict=True)
            header = next(reader, [])
            if not header or len(set(header)) != len(header):
                raise ValueError("CSV表头不能为空或重复")
            if set(header) - CSV_FIELDS:
                raise ValueError("CSV含未知列，请使用模板中的列名")
            time_columns = set(header) & {"time_seconds", "timestamp"}
            if len(time_columns) != 1:
                raise ValueError("CSV须且只能包含time_seconds或timestamp时间列")
            if "time_seconds" in header and time_unit != "relative_seconds":
                raise ValueError("time_seconds列仅允许relative_seconds；绝对时间请使用timestamp")
            if "stage" in header and "stage_duration_seconds" not in header:
                raise ValueError("存在stage列时必须提供stage_duration_seconds")
            if "stage_duration_seconds" in header and "stage" not in header:
                raise ValueError("stage_duration_seconds必须与stage列同时提供")
            if not (set(header) & {"heart_rate_bpm", "stage", "accel_x_g", "accel_y_g", "accel_z_g"}):
                raise ValueError("CSV至少需要一种信号列")
            signals = {name: [] for name in ("heart_rate", "stages", "acceleration")}
            for index, cells in enumerate(reader, 1):
                if index > MAX_ROWS:
                    raise ValueError("CSV不得超过200000行")
                if len(cells) != len(header):
                    raise ValueError(f"CSV第{index + 1}行列数与表头不一致")
                row = dict(zip(header, cells))
                timestamp = row[next(iter(time_columns))]
                if "heart_rate_bpm" in row:
                    signals["heart_rate"].append({"time": timestamp, "value": row["heart_rate_bpm"]})
                if row.get("stage", "") != "":
                    signals["stages"].append({"time": timestamp, "stage": row["stage"], "duration_seconds": row["stage_duration_seconds"]})
                elif row.get("stage_duration_seconds", "") != "":
                    raise ValueError(f"CSV第{index + 1}行阶段名称不能为空")
                if set(row) & {"accel_x_g", "accel_y_g", "accel_z_g"}:
                    signals["acceleration"].append({"time": timestamp, **{axis: row.get("accel_" + axis, None) for axis in ("x_g", "y_g", "z_g")}})
        except csv.Error:
            raise ValueError("CSV格式无效，请检查引号和字段长度") from None
    else:
        raise ValueError("仅支持text/csv或application/json文件")

    origin = None
    if time_unit != "relative_seconds" and any(signals.values()):
        if start_time is None:
            raise ValueError("绝对时序必须先设置记录的带时区start_time")
        origin = _iso(start_time, "start_time")

    def relative(row, field):
        if "time" in row and "time_seconds" in row:
            raise ValueError(f"{field}的time与time_seconds不能同时提供")
        if "time_seconds" in row and time_unit != "relative_seconds":
            raise ValueError("time_seconds仅允许relative_seconds；绝对时间请使用time")
        raw = row.get("time", row.get("time_seconds"))
        if time_unit == "iso8601":
            result = (_iso(raw, field + ".time") - origin).total_seconds()
        else:
            result = _number(raw, field + ".time")
            if time_unit != "relative_seconds":
                result = result / (1000 if time_unit == "epoch_milliseconds" else 1) - origin.timestamp()
        if not math.isfinite(result):
            raise ValueError(f"{field}相对时间超出有效范围")
        return result

    for signal, rows in signals.items():
        if kind not in {"combined", signal} and rows:
            raise ValueError(f"kind={kind}不能包含{signal}信号")
        last = None
        previous_end = None
        before = after = 0
        for index, row in enumerate(rows):
            field = f"{signal}[{index}]"
            allowed = {"time", "time_seconds"} | ({"value", "bpm"} if signal == "heart_rate" else {"stage", "duration_seconds"} if signal == "stages" else {"x_g", "y_g", "z_g"})
            _keys(row, allowed, field)
            time = relative(row, field)
            if last is not None and time <= last:
                raise ValueError(f"{field}时间须严格递增，禁止重复或倒序")
            last = time
            item = {"time_seconds": time}
            if signal == "heart_rate":
                if "value" in row and "bpm" in row:
                    raise ValueError(f"{field}的value与bpm不能同时提供")
                if not (set(row) & {"value", "bpm"}):
                    raise ValueError(f"{field}缺少value或bpm")
                value = _number(row.get("value", row.get("bpm")), field + ".value", True)
                if value is not None and value <= 0:
                    raise ValueError(f"{field}心率必须为正数")
                item["value"] = value
            elif signal == "stages":
                stage = row.get("stage")
                if not isinstance(stage, str) or stage not in STAGES:
                    raise ValueError(f"{field}阶段仅支持W/N1/N2/N3/REM/Unknown，不猜测数字编码")
                duration = _number(row.get("duration_seconds"), field + ".duration_seconds")
                if duration <= 0 or not math.isfinite(time + duration):
                    raise ValueError(f"{field}阶段时长必须为有限正数")
                if previous_end is not None and time < previous_end:
                    raise ValueError(f"{field}阶段区间不可重叠")
                previous_end = time + duration
                item.update(duration_seconds=duration, stage=stage)
            else:
                if not (set(row) & {"x_g", "y_g", "z_g"}):
                    raise ValueError(f"{field}至少提供一个g单位坐标字段")
                item.update({axis: _number(row.get(axis), field + "." + axis, True) for axis in ("x_g", "y_g", "z_g")})
            before += time < 0
            after += audio_duration is not None and (time > audio_duration or (signal == "stages" and previous_end > audio_duration))
            output[signal].append(item)
        if before or after:
            output["warnings"].append(f"{signal}范围外观察计数：音频开始前{before}条，音频结束后{after}条；保留原始观察，未裁剪或自动同步。" if audio_duration is not None else f"{signal}负相对时间{before}条；已保留，未自动同步。")
    if kind != "combined" and kind != "summary" and output["summary"]:
        raise ValueError(f"kind={kind}不能包含summary，请使用combined")
    if not any(output[name] for name in ("heart_rate", "stages", "acceleration", "summary")):
        raise ValueError("导入内容没有时序观察或summary")
    output["warnings"].insert(0, "用户导入的可穿戴观察及阶段标签，来源未经NightLab验证，不是PSG参考真值；summary保留原始字段含义，不转为健康评分。")
    if synthetic:
        output["warnings"].append("此文件明确标记为synthetic合成示例，不是真人记录。")
    return output


def template_csv() -> bytes:
    return ("# synthetic: true\n"
            "time_seconds,heart_rate_bpm,stage,stage_duration_seconds,accel_x_g,accel_y_g,accel_z_g\n"
            "0,65,W,30,0.01,0.02,1\n"
            "15,,,,,,\n"
            "30,62,N1,30,0.02,,0.99\n").encode("utf-8")


def template_json() -> bytes:
    return json.dumps({"time_unit": "relative_seconds", "meta": {"synthetic": True,
        "description": "合成模板，非真人；心率缺口保留null，阶段30秒，加速度单位g。", "acceleration_unit": "g"},
        "heart_rate": [{"time_seconds": 0, "value": 65}, {"time_seconds": 15, "value": None}, {"time_seconds": 30, "value": 62}],
        "stages": [{"time_seconds": 0, "stage": "W", "duration_seconds": 30}, {"time_seconds": 30, "stage": "N1", "duration_seconds": 30}],
        "acceleration": [{"time_seconds": 0, "x_g": .01, "y_g": .02, "z_g": 1}, {"time_seconds": 15, "x_g": None, "y_g": None, "z_g": None}, {"time_seconds": 30, "x_g": .02, "y_g": None, "z_g": .99}],
        "summary": {}}, ensure_ascii=False, indent=2).encode("utf-8")
