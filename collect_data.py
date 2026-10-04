"""Collect event-aligned IMU gesture examples from the GesturePod ESP32.

The firmware must stream CSV rows in this format:
    t_ms,ax,ay,az,gx,gy,gz

Positive examples are saved as one event-centered window with rel_t_ms=0 at
the detected motion onset. Negative recordings are saved as timestamp-based
non-overlapping clips. The script uses only pyserial beyond the Python stdlib.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
import math
import os
import statistics
import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


BAUD_RATE = 115200
SAMPLE_HZ = 200
DEFAULT_PRE_S = 0.25
DEFAULT_POST_S = 1.25
DEFAULT_BASELINE_S = 0.40
DEFAULT_TRIGGER_DPS = 45.0
DEFAULT_GYRO_RANGE_DPS = 250.0
NEGATIVE_CLIP_S = 2.0
UINT32 = 1 << 32


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("pyserial is required. Install it with: python -m pip install pyserial")


def list_ports() -> None:
    require_pyserial()
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available ports:")
    for port in ports:
        print(f"  {port.device}  —  {port.description}")


def open_serial(port: str):
    require_pyserial()
    ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    # Opening a USB serial port may reset the ESP32. Let it boot, then discard
    # startup text and old queued samples before beginning a recording.
    time.sleep(2)
    ser.reset_input_buffer()
    return ser


def parse_line(raw_line: bytes | str) -> tuple[int, float, float, float, float, float, float] | None:
    """Parse a firmware data row; ignore headers and diagnostic messages."""
    if isinstance(raw_line, bytes):
        raw_line = raw_line.decode("ascii", errors="ignore")
    parts = raw_line.strip().split(",")
    if len(parts) != 7:
        return None
    try:
        timestamp = int(parts[0])
        values = tuple(float(value) for value in parts[1:])
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return (timestamp, *values)


def elapsed_ms(timestamp: int, origin: int) -> int:
    """Signed millis() difference that remains correct across uint32 rollover."""
    return ((timestamp - origin + UINT32 // 2) % UINT32) - UINT32 // 2


def read_sample(ser):
    return parse_line(ser.readline())


def gyro_magnitude_from_bias(row, bias: tuple[float, float, float]) -> float:
    return math.sqrt(sum((row[index] - bias[index - 4]) ** 2 for index in (4, 5, 6)))


def capture_gesture(
    ser,
    *,
    pre_s: float,
    post_s: float,
    baseline_s: float,
    trigger_dps: float,
    trigger_timeout_s: float,
) -> tuple[list[tuple[int, float, float, float, float, float, float]], int]:
    """Wait for a gyro onset and return pre/post samples plus its timestamp."""
    baseline_count = max(10, round(baseline_s * SAMPLE_HZ))
    pre_count = max(1, round(pre_s * SAMPLE_HZ))
    ring: deque[tuple[int, float, float, float, float, float, float]] = deque(maxlen=pre_count + 8)

    print(f"  Stay still for {baseline_s:.1f}s while gyro baseline is measured...")
    baseline = []
    deadline = time.monotonic() + baseline_s + 2.0
    while len(baseline) < baseline_count and time.monotonic() < deadline:
        row = read_sample(ser)
        if row is not None:
            baseline.append(row)
            ring.append(row)
    if len(baseline) < max(5, baseline_count // 2):
        raise RuntimeError(f"Only received {len(baseline)} baseline samples; check the serial connection.")

    bias = tuple(statistics.mean(row[i] for row in baseline) for i in (4, 5, 6))
    print(f"  Now perform the gesture once (trigger: {trigger_dps:g} deg/s)...")
    deadline = time.monotonic() + trigger_timeout_s
    candidate_time = None
    candidate_count = 0
    trigger_time = None
    while time.monotonic() < deadline:
        row = read_sample(ser)
        if row is None:
            continue
        ring.append(row)
        if gyro_magnitude_from_bias(row, bias) >= trigger_dps:
            if candidate_count == 0:
                candidate_time = row[0]
            candidate_count += 1
            if candidate_count >= 2:
                trigger_time = candidate_time
                break
        else:
            candidate_count = 0
            candidate_time = None

    if trigger_time is None:
        raise TimeoutError("No movement trigger detected. Try a stronger motion or lower --trigger-dps.")

    # Keep the recent pre-trigger history, then continue until the requested
    # post-trigger duration has elapsed according to the sensor timestamp.
    rows = list(ring)
    deadline = time.monotonic() + post_s + 2.0
    target_post_ms = round(post_s * 1000)
    while time.monotonic() < deadline:
        if rows and elapsed_ms(rows[-1][0], trigger_time) >= target_post_ms:
            break
        row = read_sample(ser)
        if row is not None:
            rows.append(row)
    if not rows or elapsed_ms(rows[-1][0], trigger_time) < target_post_ms - 20:
        raise RuntimeError("Recording ended before the full post-gesture window was captured.")
    return rows, trigger_time


def aligned_rows(rows, trigger_time: int):
    """Return the stable CSV schema with zero at the detected event onset."""
    return [(elapsed_ms(row[0], trigger_time), *row[1:]) for row in rows]


def save_rows(rows, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["rel_t_ms", "ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(rows)


def quality_report(rows, *, gyro_range_dps: float, expected_duration_s: float) -> list[str]:
    """Return data quality warnings without silently rejecting a labeled take."""
    warnings = []
    if len(rows) < expected_duration_s * SAMPLE_HZ * 0.70:
        warnings.append(f"short capture ({len(rows)} rows)")

    times = [row[0] for row in rows]
    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    positive_intervals = [dt for dt in intervals if dt > 0]
    if positive_intervals:
        median_dt = statistics.median(positive_intervals)
        gaps = sum(dt > max(12, median_dt * 2.5) for dt in intervals)
        if median_dt < 3 or median_dt > 8:
            warnings.append(f"unexpected median sample interval ({median_dt:g} ms; target is about 5 ms)")
        if gaps:
            warnings.append(f"{gaps} sample gap(s) greater than 2.5× median interval")

    clip_limit = gyro_range_dps * 0.98
    clipped = sum(any(abs(row[i]) >= clip_limit for i in (4, 5, 6)) for row in rows)
    if clipped:
        warnings.append(f"gyro near configured ±{gyro_range_dps:g} deg/s limit in {clipped} row(s)")

    # Count separated above-threshold bouts as a review cue. Ringing can split
    # a single physical tap, so this is advisory rather than a pass/fail test.
    moving = [i for i, row in enumerate(rows) if math.sqrt(sum(row[j] ** 2 for j in (4, 5, 6))) >= 70]
    bouts = []
    for index in moving:
        if not bouts or times[index] - times[bouts[-1][-1]] > 200:
            bouts.append([index])
        else:
            bouts[-1].append(index)
    if len(bouts) > 3:
        warnings.append(f"{len(bouts)} separated high-gyro bouts; review whether this was one clean gesture")
    return warnings


def unique_csv_path(root: str, label: str, suffix: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{label}_{stamp}{suffix}.csv"
    return os.path.join(root, label, base)


def cmd_gesture(args) -> None:
    ser = open_serial(args.port)
    try:
        print(f"\nCollecting '{args.label}' — {args.reps} event-aligned examples.")
        print("Each take waits for a still baseline, detects movement onset, and captures pre/post samples.")
        rep = 0
        while rep < args.reps:
            print(f"\n[{rep + 1}/{args.reps}] Get ready. Keep still during the countdown.")
            ser.reset_input_buffer()
            for count in range(args.countdown, 0, -1):
                print(f"  {count}...", end="", flush=True)
                time.sleep(1)
            print(" GO")
            try:
                raw_rows, trigger_time = capture_gesture(
                    ser,
                    pre_s=args.pre,
                    post_s=args.post,
                    baseline_s=args.baseline,
                    trigger_dps=args.trigger_dps,
                    trigger_timeout_s=args.trigger_timeout,
                )
            except (RuntimeError, TimeoutError) as exc:
                print(f"  Capture issue: {exc}")
                choice = input("  Enter to retry, or 'q' to stop: ").strip().lower()
                if choice == "q":
                    break
                continue

            rows = aligned_rows(raw_rows, trigger_time)
            expected_duration = args.pre + args.post
            warnings = quality_report(
                rows,
                gyro_range_dps=args.gyro_range,
                expected_duration_s=expected_duration,
            )
            print(f"  Captured {len(rows)} samples; event-aligned window {rows[0][0]} to {rows[-1][0]} ms.")
            if warnings:
                for warning in warnings:
                    print(f"  REVIEW: {warning}")
            else:
                print("  Quality checks passed (still review the waveform and label).")

            choice = input("  Enter to save, 'r' to redo, 'q' to stop: ").strip().lower()
            if choice == "q":
                break
            if choice == "r":
                continue
            path = unique_csv_path(args.out, args.label)
            save_rows(rows, path)
            print(f"  Saved {path}")
            rep += 1
    finally:
        ser.close()
    print(f"\nFinished. Accepted {rep} example(s) in {os.path.join(args.out, args.label)}")


def record_for(ser, duration_s: float):
    """Read timestamped samples for a wall-clock duration."""
    rows = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        row = read_sample(ser)
        if row is not None:
            rows.append(row)
    return rows


def cmd_negative(args) -> None:
    ser = open_serial(args.port)
    try:
        print(f"\nRecording {args.duration:g}s of negative/non-gesture cane use.")
        print("Include normal walking, cane placement, single taps, and partial gestures.")
        input("Press Enter when ready to start...")
        ser.reset_input_buffer()
        print("Recording...")
        rows = record_for(ser, args.duration)
        print(f"Captured {len(rows)} samples.")
        if not rows:
            print("No data received — check the serial connection.")
            return

        clip_ms = round(args.clip_seconds * 1000)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        clip, clip_start, clip_number = [], rows[0][0], 0
        for row in rows:
            if elapsed_ms(row[0], clip_start) >= clip_ms:
                if clip:
                    path = unique_csv_path(args.out, "negative", f"_{timestamp}_{clip_number:03d}")
                    save_rows([(elapsed_ms(sample[0], clip_start), *sample[1:]) for sample in clip], path)
                    clip_number += 1
                clip, clip_start = [], row[0]
            clip.append(row)
        # Save only complete negative clips so every file has comparable duration.
        if clip and elapsed_ms(clip[-1][0], clip_start) >= clip_ms - 10:
            path = unique_csv_path(args.out, "negative", f"_{timestamp}_{clip_number:03d}")
            save_rows([(elapsed_ms(sample[0], clip_start), *sample[1:]) for sample in clip], path)
            clip_number += 1
        print(f"Saved {clip_number} complete {args.clip_seconds:g}s negative clips to {os.path.join(args.out, 'negative')}")
    finally:
        ser.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Event-aligned GesturePod IMU data collector")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-ports", help="List serial ports")

    gesture = sub.add_parser("gesture", help="Collect event-aligned positive gesture examples")
    gesture.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    gesture.add_argument("--label", required=True, help="Gesture label, e.g. double_tap")
    gesture.add_argument("--reps", type=int, default=20)
    gesture.add_argument("--out", default="dataset", help="Output dataset root")
    gesture.add_argument("--countdown", type=int, default=2, help="Stillness countdown in seconds")
    gesture.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_S, help="Still gyro calibration duration")
    gesture.add_argument("--pre", type=float, default=DEFAULT_PRE_S, help="Seconds to retain before detected onset")
    gesture.add_argument("--post", type=float, default=DEFAULT_POST_S, help="Seconds to capture after detected onset")
    gesture.add_argument("--trigger-dps", type=float, default=DEFAULT_TRIGGER_DPS, help="Gyro movement trigger after baseline bias removal")
    gesture.add_argument("--trigger-timeout", type=float, default=8.0, help="Seconds to wait for gesture onset")
    gesture.add_argument("--gyro-range", type=float, default=DEFAULT_GYRO_RANGE_DPS, help="Configured firmware gyro range for clipping checks")

    negative = sub.add_parser("negative", help="Collect continuous negative cane-use examples")
    negative.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    negative.add_argument("--duration", type=float, default=480.0, help="Recording duration in seconds")
    negative.add_argument("--clip-seconds", type=float, default=NEGATIVE_CLIP_S)
    negative.add_argument("--out", default="dataset", help="Output dataset root")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "list-ports":
        list_ports()
    elif args.command == "gesture":
        if args.reps < 1 or args.countdown < 0 or args.pre < 0 or args.post <= 0:
            parser.error("reps must be positive; countdown/pre must be non-negative; post must be positive")
        cmd_gesture(args)
    elif args.command == "negative":
        if args.duration <= 0 or args.clip_seconds <= 0:
            parser.error("duration and clip-seconds must be positive")
        cmd_negative(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
