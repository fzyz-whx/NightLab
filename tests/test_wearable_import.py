import json
import unittest

from nightlab.services.wearable_import import (
    MAX_BYTES, parse_wearable, template_csv, template_json,
)


def parse_json(payload, **kwargs):
    return parse_wearable(json.dumps(payload).encode(), "application/json", **kwargs)


class WearableImportTests(unittest.TestCase):
    def test_templates_equivalent_and_synthetic_with_gaps(self):
        csv = parse_wearable(template_csv(), "text/csv; charset=utf-8")
        obj = parse_wearable(template_json(), "application/json")
        self.assertEqual(csv, obj)
        self.assertIsNone(csv["heart_rate"][1]["value"])
        self.assertIsNone(csv["acceleration"][1]["x_g"])
        self.assertEqual(csv["stages"][0]["duration_seconds"], 30)
        self.assertTrue(any("合成" in warning for warning in csv["warnings"]))

    def test_real_bytes_bom_and_alias_equivalence(self):
        csv = b"\xef\xbb\xbftime_seconds,heart_rate_bpm,accel_z_g\r\n0,60,1\r\n1,,\r\n"
        expected = parse_json({"heart_rate": [{"time": 0, "bpm": 60}, {"time": 1, "bpm": None}],
                               "acceleration": [{"time": 0, "z_g": 1}, {"time": 1, "z_g": None}]})
        self.assertEqual(parse_wearable(csv, "text/csv"), expected)

    def test_absolute_seconds_milliseconds_iso_across_midnight(self):
        start = "2026-10-25T23:59:59+08:00"
        for unit, raw in [("epoch_seconds", 1792944001), ("epoch_milliseconds", 1792944001000),
                          ("iso8601", "2026-10-26T00:00:01+08:00")]:
            result = parse_json({"time_unit": unit, "heart_rate": [{"time": raw, "value": 60}]}, start_time=start)
            self.assertEqual(result["heart_rate"][0]["time_seconds"], 2)
        result = parse_wearable(b"timestamp,heart_rate_bpm\n2026-10-26T00:00:01+08:00,60\n", "text/csv",
                                time_unit="iso8601", start_time=start)
        self.assertEqual(result["heart_rate"][0]["time_seconds"], 2)

    def test_missing_timezone_and_invalid_times(self):
        for start in [None, "2026-10-25T23:59:59", "bad"]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                parse_json({"time_unit": "epoch_seconds", "heart_rate": [{"time": 1, "value": 60}]}, start_time=start)
        for time in ["2026-10-26T00:00:01", "bad", None]:
            with self.subTest(time=time), self.assertRaises(ValueError):
                parse_json({"time_unit": "iso8601", "heart_rate": [{"time": time, "value": 60}]}, start_time="2026-10-26T00:00:00Z")
        with self.assertRaises(ValueError):
            parse_json({"time_unit": "epoch_seconds", "heart_rate": [{"time_seconds": 1, "value": 60}]}, start_time="2026-10-26T00:00:00Z")

    def test_negative_and_outside_observations_retained_and_counted(self):
        result = parse_json({"heart_rate": [{"time": -1, "value": None}, {"time": 101, "value": 60}],
                             "stages": [{"time": 90, "duration_seconds": 30, "stage": "Unknown"}]}, audio_duration=100)
        self.assertEqual(len(result["heart_rate"]), 2)
        self.assertEqual(result["stages"][0]["stage"], "Unknown")
        self.assertTrue(any("开始前1条，音频结束后1条" in value for value in result["warnings"]))
        self.assertTrue(any("stages范围外" in value for value in result["warnings"]))

    def test_nonfinite_or_nonpositive_hr_and_invalid_axes_rejected(self):
        for value in ["NaN", "Infinity", float("inf"), -1, 0, True, " ", [], {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_json({"heart_rate": [{"time": 0, "value": value}]})
        for value in ["NaN", "Infinity", True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_json({"acceleration": [{"time": 0, "x_g": value}]})

    def test_stage_labels_duration_overlap_and_order(self):
        for label in [0, "0", "deep", "unknown", None]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_json({"stages": [{"time": 0, "stage": label, "duration_seconds": 30}]})
        for duration in [None, 0, -1, "NaN"]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                parse_json({"stages": [{"time": 0, "stage": "W", "duration_seconds": duration}]})
        with self.assertRaisesRegex(ValueError, "重叠"):
            parse_json({"stages": [{"time": 0, "stage": "W", "duration_seconds": 30},
                                   {"time": 29, "stage": "N1", "duration_seconds": 30}]})
        for signal, fields in [("heart_rate", {"value": 60}), ("stages", {"stage": "W", "duration_seconds": 1}),
                               ("acceleration", {"x_g": None})]:
            for times in [(1, 1), (1, 0)]:
                with self.subTest(signal=signal, times=times), self.assertRaisesRegex(ValueError, "严格递增"):
                    parse_json({signal: [{"time": time, **fields} for time in times]})

    def test_csv_headers_missing_columns_and_bad_rows(self):
        for csv in ["time_seconds,heart_rate_bpm,heart_rate_bpm\n0,60,61", "heart_rate_bpm\n60",
                    "time_seconds,stage\n0,W", "time_seconds,mystery\n0,1", "time_seconds\n0",
                    "time_seconds,timestamp,heart_rate_bpm\n0,0,60", "time_seconds,heart_rate_bpm\n0,60,1",
                    "time_seconds,stage,stage_duration_seconds\n0,,30"]:
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                parse_wearable(csv.encode(), "text/csv")

    def test_summary_only_has_no_fabricated_curves(self):
        result = parse_json({"summary": {"vendor_sleep_score": 78, "note": "vendor estimate", "estimated": True}})
        absolute_summary = parse_json({"time_unit": "iso8601", "summary": {"vendor_sleep_score": 78}})
        self.assertEqual(absolute_summary["heart_rate"], [])
        for signal in ["heart_rate", "stages", "acceleration"]:
            self.assertEqual(result[signal], [])
        self.assertEqual(result["summary"]["vendor_sleep_score"], 78)
        for summary in [{"a": []}, {"a": float("nan")}, {"a": "x" * 1001}]:
            with self.subTest(summary=summary), self.assertRaises(ValueError):
                parse_json({"summary": summary})

    def test_limits_invalid_json_unknown_fields_and_kinds(self):
        with self.assertRaisesRegex(ValueError, "20MiB"):
            parse_wearable(b"x" * (MAX_BYTES + 1), "text/csv")
        with self.assertRaises(ValueError):
            parse_wearable(b"\xff", "text/csv")
        for raw in [b'{"summary":{},"summary":{}}', b'{', b'[]', b'{"unknown":1}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_wearable(raw, "application/json")
        for payload in [{"heart_rate": [{"time": 0, "value": 60, "extra": 1}]},
                        {"heart_rate": [{"time": 0, "value": 60, "bpm": 60}]},
                        {"meta": {"acceleration_unit": "m/s2"}, "summary": {"a": 1}}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_json(payload)
        with self.assertRaises(ValueError):
            parse_json({"heart_rate": [{"time": 0, "value": 60}]}, kind="stages")
        with self.assertRaisesRegex(ValueError, "200000"):
            parse_json({"heart_rate": [{"time": 0, "value": 60}] * 200001})


if __name__ == "__main__":
    unittest.main()
