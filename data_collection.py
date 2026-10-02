"""
GesturePod-style Cane — Data Collection Tool

Talks to the ESP32 over Serial (the same CSV stream you saw in the
Serial Monitor: t_ms,ax,ay,az,gx,gy,gz) and helps you build a labeled
dataset for training, following the collection protocol from the
GesturePod paper (Section 4.3):

  1. Positive examples ("gesture" mode): the script cues a gesture by
     name, counts down, records a fixed window, and saves it as its
     own CSV. Repeat across sessions with different grip/handedness/
     flooring for robustness, same as the paper's volunteers did.

  2. Negative examples ("negative" mode): just walk around with the
     cane normally (mix in some partial gestures on purpose — half a
     twist, a single tap — since the paper found these are the main
     source of false positives). The script records continuously and
     auto-chops the stream into 2-second clips, same duration as the
     positive examples.

Each saved example is one CSV file with columns:
    rel_t_ms,ax,ay,az,gx,gy,gz
where rel_t_ms is zeroed to the start of that example (0, 5, 10, ...ms).

Folder layout produced:
    dataset/
      double_tap/double_tap_20260904_141002.csv
      right_twist/right_twist_20260904_141115.csv
      ...
      negative/negative_20260904_141500_000.csv
      negative/negative_20260904_141500_001.csv
      ...

Usage:
    pip install pyserial

    python collect_data.py list-ports

    python collect_data.py gesture --port COM5 --label right_twist --reps 20
    python collect_data.py gesture --port COM5 --label double_tap --reps 20
    (repeat for: left_twist, twirl, double_swipe)

    python collect_data.py negative --port COM5 --duration 480
    (480s = 8 minutes, matching the paper's negative collection run;
     run this multiple times across different sessions/environments)
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import serial
import serial.tools.list_ports

BAUD_RATE = 115200
GESTURE_WINDOW_S = 2.5   # a little wider than the paper's 2s window, trimmed later if needed
NEGATIVE_CLIP_S = 2.0    # matches the paper's 2s negative clip length


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available ports:")
    for p in ports:
        print(f"  {p.device}  —  {p.description}")


def open_serial(port):
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    time.sleep(2)  # let the ESP32 finish any boot-time reset the serial connection triggers
    ser.reset_input_buffer()
    return ser


def parse_line(raw_line):
    """Returns (t_ms, ax, ay, az, gx, gy, gz) or None if the line isn't a data row
    (e.g. the header line, or a [diag] message)."""
    parts = raw_line.strip().split(",")
    if len(parts) != 7:
        return None
    try:
        t_ms = int(parts[0])
        vals = [float(x) for x in parts[1:]]
        return (t_ms, *vals)
    except ValueError:
        return None


def record_for(ser, duration_s):
    """Reads from serial for duration_s wall-clock seconds, returns a list of
    parsed rows with rel_t_ms zeroed to the first sample received."""
    rows = []
    first_t_ms = None
    deadline = time.time() + duration_s
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore")
        if not line:
            continue
        parsed = parse_line(line)
        if parsed is None:
            continue
        t_ms = parsed[0]
        if first_t_ms is None:
            first_t_ms = t_ms
        rel_t_ms = t_ms - first_t_ms
        rows.append((rel_t_ms, *parsed[1:]))
    return rows


def save_rows(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_t_ms", "ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(rows)


def cmd_gesture(args):
    ser = open_serial(args.port)
    out_root = args.out
    label = args.label

    print(f"\nCollecting '{label}' — {args.reps} reps, {GESTURE_WINDOW_S}s window each.")
    print("For each rep: get ready during the countdown, perform the gesture once")
    print("when you see GO, keep the cane still for the rest of the window.\n")

    i = 0
    while i < args.reps:
        print(f"[{i+1}/{args.reps}] Get ready...", end="", flush=True)
        time.sleep(1.2)
        print(" GO!")
        ser.reset_input_buffer()  # drop anything queued up during the countdown
        rows = record_for(ser, GESTURE_WINDOW_S)

        if len(rows) < GESTURE_WINDOW_S * 200 * 0.5:
            print(f"  -> Only got {len(rows)} samples (expected ~{int(GESTURE_WINDOW_S*200)}). "
                  f"Check the serial connection. Retrying this rep.")
            continue

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_root, label, f"{label}_{ts}.csv")
        save_rows(rows, path)
        print(f"  -> saved {len(rows)} samples to {path}")

        choice = input("  Press Enter to continue, 'r' + Enter to redo this rep, 'q' + Enter to stop: ").strip().lower()
        if choice == "r":
            os.remove(path)
            continue  # redo same i
        if choice == "q":
            break
        i += 1

    ser.close()
    print(f"\nDone. Collected examples are in {os.path.join(out_root, label)}")


def cmd_negative(args):
    ser = open_serial(args.port)
    out_root = args.out

    print(f"\nRecording {args.duration}s of negative (non-gesture) cane use.")
    print("Walk around normally with the cane. Mix in some partial gestures on")
    print("purpose (half a twist, a single tap) — these are the main source of")
    print("false positives, so the model needs to see them labeled as 'negative'.\n")
    input("Press Enter when ready to start...")

    ser.reset_input_buffer()
    print("Recording...")
    rows = record_for(ser, args.duration)
    print(f"Captured {len(rows)} samples.")

    if not rows:
        print("No data received — check the serial connection.")
        ser.close()
        return

    # Chop the continuous stream into non-overlapping 2s clips based on rel_t_ms,
    # so negative examples are stored the same shape as positive examples.
    clip_samples = int(NEGATIVE_CLIP_S * 200)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_clips = 0
    for start in range(0, len(rows) - clip_samples + 1, clip_samples):
        clip = rows[start:start + clip_samples]
        base_t = clip[0][0]
        clip = [(r[0] - base_t, *r[1:]) for r in clip]  # re-zero rel_t_ms per clip
        path = os.path.join(out_root, "negative", f"negative_{ts}_{n_clips:03d}.csv")
        save_rows(clip, path)
        n_clips += 1

    ser.close()
    print(f"\nDone. Saved {n_clips} negative clips to {os.path.join(out_root, 'negative')}")


def main():
    parser = argparse.ArgumentParser(description="GesturePod cane data collection tool")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-ports", help="List available serial ports")

    p_gesture = sub.add_parser("gesture", help="Collect cued positive examples for one gesture")
    p_gesture.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    p_gesture.add_argument("--label", required=True,
                            help="Gesture name, e.g. double_tap, right_twist, left_twist, twirl, double_swipe")
    p_gesture.add_argument("--reps", type=int, default=20, help="Number of examples to collect (default 20)")
    p_gesture.add_argument("--out", default="dataset", help="Output dataset root folder (default: dataset)")

    p_negative = sub.add_parser("negative", help="Collect continuous negative (non-gesture) examples")
    p_negative.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    p_negative.add_argument("--duration", type=float, default=480,
                             help="Recording duration in seconds (default 480 = 8 min, matching the paper)")
    p_negative.add_argument("--out", default="dataset", help="Output dataset root folder (default: dataset)")

    args = parser.parse_args()

    if args.command == "list-ports":
        list_ports()
    elif args.command == "gesture":
        cmd_gesture(args)
    elif args.command == "negative":
        cmd_negative(args)


if __name__ == "__main__":
    main()