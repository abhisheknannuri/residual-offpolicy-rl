#!/usr/bin/env python3
"""
Identify the VEC Infinity-3 foot pedal button codes (Step 1 of pedal integration).

Auto-finds the VEC foot pedal on /dev/input, then listens for a fixed number of
seconds and prints every key event (press / release) with its evdev code + name.
A summary at the end lists each distinct code and how many times it was pressed,
so you can map physical LEFT / MIDDLE / RIGHT pedals to their codes.

Debug
-----
    Run this to see which device event the pedal uses:
        ls -l /dev/input/by-id/ | grep -i vec
        lrwxrwxrwx - root  7 Aug 16:30 usb-VEC_VEC_USB_Footpedal-event-if00 -> ../event7
    
    The output device event number (7 in this example) should be passed to the --device argument:
        python identify_pedal.py --device /dev/input/event7

Usage
-----
    # Needs read access to /dev/input/event* (be in the 'input' group, or sudo):
    python identify_pedal.py                 # listens for 25 s
    python identify_pedal.py --seconds 40
    python identify_pedal.py --device /dev/input/event12

Suggested procedure
-------------------
    Press LEFT pedal twice, pause, MIDDLE twice, pause, RIGHT twice.
    The ordered output makes the mapping unambiguous.

Ctrl+C quits early.
"""

import argparse
import select
import sys
import time

import evdev
from evdev import ecodes


PEDAL_NAME_HINTS = ("footpedal", "foot pedal", "vec")


def find_pedal(explicit_path=None):
    """Return an evdev.InputDevice for the pedal, or None."""
    if explicit_path:
        return evdev.InputDevice(explicit_path)
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except Exception:  # noqa: BLE001 - skip devices we can't open
            continue
        name = (dev.name or "").lower()
        if any(hint in name for hint in PEDAL_NAME_HINTS):
            return dev
    return None


def code_name(code):
    """Human-readable name for a key/button code (may be a list)."""
    name = ecodes.bytype.get(ecodes.EV_KEY, {}).get(code)
    if isinstance(name, (list, tuple)):
        name = "/".join(name)
    return name or f"code_{code}"


def main():
    parser = argparse.ArgumentParser(description="Identify VEC foot pedal button codes.")
    parser.add_argument("--seconds", type=float, default=25.0,
                        help="How long to listen (default 25 s).")
    parser.add_argument("--device", type=str, default=None,
                        help="Explicit /dev/input/eventN path (else auto-detect).")
    args = parser.parse_args()

    try:
        dev = find_pedal(args.device)
    except PermissionError:
        print("Permission denied opening the device. Add yourself to the 'input' "
              "group (sudo usermod -aG input $USER; then re-login or 'newgrp input'), "
              "or run this with sudo.")
        sys.exit(1)

    if dev is None:
        print("No VEC foot pedal found. Available input devices:")
        for path in evdev.list_devices():
            try:
                print(f"    {path}  ->  {evdev.InputDevice(path).name}")
            except Exception:  # noqa: BLE001
                print(f"    {path}  ->  <cannot open>")
        print("\nPass one explicitly with --device /dev/input/eventN if needed.")
        sys.exit(1)

    print("=" * 68)
    print(f"Reading pedal: {dev.path}   ({dev.name})")
    print(f"Listening for {args.seconds:.0f} s. Press each pedal now.")
    print("Suggested: LEFT x2, pause, MIDDLE x2, pause, RIGHT x2.")
    print("=" * 68)

    press_counts = {}
    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            r, _, _ = select.select([dev.fd], [], [], min(0.5, remaining))
            if not r:
                continue
            for event in dev.read():
                if event.type != ecodes.EV_KEY:
                    continue
                name = code_name(event.code)
                if event.value == 1:      # key down
                    press_counts[event.code] = press_counts.get(event.code, 0) + 1
                    print(f"  [{time.strftime('%H:%M:%S')}]  PRESS    "
                          f"code={event.code:<5} name={name}")
                elif event.value == 0:    # key up
                    print(f"  [{time.strftime('%H:%M:%S')}]  release  "
                          f"code={event.code:<5} name={name}")
    except KeyboardInterrupt:
        print("\n(stopped early)")

    print("\n" + "=" * 68)
    print("SUMMARY - distinct pedals seen (by press count):")
    if not press_counts:
        print("  (no presses detected)")
    else:
        for code in sorted(press_counts):
            print(f"  code={code:<5} name={code_name(code):<16} presses={press_counts[code]}")
    print("=" * 68)
    print("Tell me which physical pedal (left/middle/right) produced each code.")


if __name__ == "__main__":
    main()
