import pretty_midi


def make_midi(path, n_notes=60, program=0, base=60):
    """Synthetic test MIDI (a scale pattern). Used only as a test fixture."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    t = 0.0
    for i in range(n_notes):
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=base + (i * 2) % 12, start=t, end=t + 0.5))
        t += 0.5
    pm.instruments.append(inst)
    pm.write(str(path))
    return path
