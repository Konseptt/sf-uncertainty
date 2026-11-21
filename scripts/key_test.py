"""Keyboard check: opens a small window, reports every key PsychoPy receives for 15 s.

    .venv-psychopy/bin/python scripts/key_test.py [event|ptb|iohub]
"""
import sys
import time

from psychopy import visual
from psychopy.hardware import keyboard

backend = sys.argv[1] if len(sys.argv) > 1 else "event"
win = visual.Window((600, 300), units="pix", color=0)
try:
    win.winHandle.activate()
except AttributeError:
    pass
kb = keyboard.Keyboard(backend=backend)
print("requested backend:", backend, "| active:", keyboard.KeyboardDevice._backend, flush=True)
msg = visual.TextStim(win, "Press keys (15 s)", color=-1, height=24)
t0 = time.time()
while time.time() - t0 < 15:
    msg.draw()
    win.flip()
    for k in kb.getKeys(waitRelease=False):  # 'event' backend: names arrive as [name, t]
        print(f"got key: {k.name}", flush=True)
        msg.text = f"got: {k.name}"
win.close()
print("done")
