"""Voice control module: Vosk-based wake-word state machine + VoiceController.

Wake-word protocol
------------------
1. Hear "robot"  → enter *awaiting* phase (timer starts)
2. Within WAKE_TIMEOUT_S hear a valid command → dispatch it, return to idle
3. Timeout or "[unk]" while awaiting → return to idle (no dispatch)
4. "stop" is always dispatched regardless of phase

Public API
----------
- GRAMMAR_WORDS : list[str]   – words fed to KaldiRecognizer grammar
- VALID_COMMANDS : set[str]   – commands that can be dispatched (no wake needed for "stop")
- parse_command(word, state, now) -> str | None  – pure state-machine function
- VoiceController               – threaded Vosk + PyAudio driver
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000
CHUNK_FRAMES: int = 4000
WAKE_TIMEOUT_S: float = 2.0

# Path to Vosk model relative to repo root (one level up from this file).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "vosk-model-small-en-us-0.15")

# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

GRAMMAR_WORDS: list = [
    "robot",
    "strawberry",
    "banana",
    "tomato",
    "all",
    "stop",
    "home",
    "refresh",
    "survey",
    "[unk]",
]

# All dispatachable commands (excludes "robot" wake word and "[unk]" noise token).
VALID_COMMANDS: set = set(GRAMMAR_WORDS) - {"robot", "[unk]"}


# ---------------------------------------------------------------------------
# Pure state-machine function
# ---------------------------------------------------------------------------

def parse_command(
    word: str,
    state: Dict[str, object],
    now: float,
) -> Optional[str]:
    """Update *state* in-place and return a command string or None.

    Parameters
    ----------
    word  : recognised word from Vosk (lower-case)
    state : mutable dict with keys ``phase`` (str) and ``wake_time`` (float)
    now   : current timestamp (seconds, e.g. time.monotonic())

    Returns
    -------
    str   – command to dispatch (e.g. ``"stop"``, ``"banana"``)
    None  – no action needed
    """
    # "stop" always wins regardless of phase.
    if word == "stop":
        state["phase"] = "idle"
        return "stop"

    # "[unk]" is never dispatched; don't change state (stays awaiting if so).
    if word == "[unk]":
        return None

    # Wake word: enter / refresh awaiting phase.
    if word == "robot":
        state["phase"] = "awaiting"
        state["wake_time"] = now
        return None

    # Any other word: only act while awaiting and within timeout.
    if state["phase"] == "awaiting":
        elapsed = now - float(state.get("wake_time", 0.0))
        if elapsed > WAKE_TIMEOUT_S:
            # Timed out — silently reset.
            state["phase"] = "idle"
            return None
        if word in VALID_COMMANDS:
            state["phase"] = "idle"
            return word

    return None


# ---------------------------------------------------------------------------
# VoiceController
# ---------------------------------------------------------------------------

class VoiceController:
    """Threaded Vosk + PyAudio voice controller.

    Parameters
    ----------
    root         : tkinter root (used for root.after() thread-safe callbacks)
    command_map  : dict mapping command strings to callables
    model_path   : path to Vosk model directory (defaults to DEFAULT_MODEL_PATH)
    status_label : optional tkinter Label to show recognition status
    """

    def __init__(
        self,
        root,
        command_map: Dict[str, Callable],
        model_path: str = DEFAULT_MODEL_PATH,
        status_label=None,
    ) -> None:
        import vosk
        import pyaudio

        vosk.SetLogLevel(-1)

        self._root = root
        self._commands = command_map
        self._label = status_label
        self._stop_event = threading.Event()

        # Load Vosk model and recogniser.
        model = vosk.Model(model_path)
        grammar_json = json.dumps(GRAMMAR_WORDS)
        self._recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE, grammar_json)

        # Open PyAudio stream.
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )

        # State dict for parse_command.
        self._state: Dict[str, object] = {"phase": "idle", "wake_time": 0.0}
        self._update_status("\U0001f3a4 Voice", "grey")

        # Start daemon listener thread.
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the listener thread to stop and release audio resources."""
        self._stop_event.set()
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass
        try:
            self._pa.terminate()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal thread
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        """Daemon thread: read audio chunks and parse recognised words."""
        was_idle = True  # track phase transition to beep only once

        while not self._stop_event.is_set():
            try:
                data = self._stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            except Exception:
                break

            now = time.monotonic()

            if self._recognizer.AcceptWaveform(data):
                result = json.loads(self._recognizer.Result())
                text = result.get("text", "").strip()
            else:
                partial = json.loads(self._recognizer.PartialResult())
                text = partial.get("partial", "").strip()

            if not text:
                # Check for timeout while awaiting (show "???" and reset label).
                if self._state["phase"] == "awaiting":
                    elapsed = now - float(self._state.get("wake_time", 0.0))
                    if elapsed > WAKE_TIMEOUT_S:
                        self._state["phase"] = "idle"
                        was_idle = True
                        self._update_status("\U0001f3a4 ???", "#cc0000")
                        self._root.after(1000, lambda: self._update_status(
                            "\U0001f3a4 Voice", "grey"))
                continue

            for word in text.split():
                cmd = parse_command(word, self._state, now)

                # Beep on first transition to awaiting.
                if self._state["phase"] == "awaiting" and was_idle:
                    self._beep()
                    self._update_status("\U0001f3a4 Listening...", "#cc8800")
                    was_idle = False
                elif self._state["phase"] == "idle":
                    was_idle = True

                if cmd is not None:
                    self._dispatch(cmd)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _beep(self) -> None:
        """Play a system beep (Windows only; silent on other platforms)."""
        try:
            import winsound
            winsound.PlaySound("SystemAsterisk",
                               winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception:
            pass

    def _dispatch(self, cmd: str) -> None:
        """Look up callback, update status label, invoke via root.after."""
        callback = self._commands.get(cmd)
        if callback is None:
            return
        self._update_status(f'\U0001f3a4 "{cmd}" ✓', "#008800")
        self._root.after(0, callback)
        self._root.after(1500, lambda: self._update_status(
            "\U0001f3a4 Voice", "grey"))

    def _update_status(self, text: str, fg: str) -> None:
        """Thread-safe label update via root.after."""
        if self._label is not None:
            self._root.after(0, lambda: self._label.configure(
                text=text, foreground=fg
            ))
