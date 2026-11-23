"""Render MIDI to WAV. Uses FluidSynth + SoundFont if available, else a built-in NumPy synth."""
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

import config


class AudioRenderError(Exception):
    pass


_SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
]


def find_soundfont() -> Optional[str]:
    env = os.environ.get("SOUNDFONT")
    for c in ([env] if env else []) + _SOUNDFONT_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


def _synth_numpy(pm: pretty_midi.PrettyMIDI, sr: int, max_seconds: float) -> np.ndarray:
    end = min(pm.get_end_time(), max_seconds) + 0.5
    audio = np.zeros(int(end * sr) + sr, dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= max_seconds:
                continue
            dur = max(min(n.end, max_seconds) - n.start, 0.05)
            t = np.arange(int((dur + 0.25) * sr)) / sr
            f = 440.0 * 2 ** ((n.pitch - 69) / 12)
            wave_ = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) + 0.25 * np.sin(6 * np.pi * f * t)
            env = np.minimum(t / 0.01, 1.0) * np.exp(-2.0 * t) * np.where(t <= dur, 1.0, np.clip(1 - (t - dur) / 0.25, 0, 1))
            s = int(n.start * sr)
            seg = (wave_ * env * (n.velocity / 127) * 0.25).astype(np.float32)
            audio[s:s + len(seg)] += seg[: len(audio) - s]
    return audio


def render_wav(midi_path, wav_path, max_seconds: float = config.MAX_AUDIO_SECONDS) -> str:
    """Write a WAV file. Returns the method used ('fluidsynth' or 'numpy-synth')."""
    sr = config.SAMPLE_RATE
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        raise AudioRenderError("Could not read MIDI file for rendering.") from exc

    audio, method = None, "numpy-synth"
    sf = find_soundfont()
    if sf:
        try:
            audio = pm.fluidsynth(fs=sr, sf2_path=sf)[: int(max_seconds * sr)].astype(np.float32)
            method = "fluidsynth"
        except Exception:
            audio = None  # fall back silently to the built-in synth
    if audio is None:
        try:
            audio = _synth_numpy(pm, sr, max_seconds)
        except Exception as exc:
            raise AudioRenderError("Audio rendering failed.") from exc
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak == 0.0:
        raise AudioRenderError("Rendered audio is silent.")
    audio = audio / peak * 0.9
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return method
