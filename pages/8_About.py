import streamlit as st

from components import sidebar

sidebar.hero("About MelodyMind AI", "Intelligent Music Composition Studio")
st.markdown("""
**Project.** MelodyMind AI is an LSTM-based melody generator built as an **Artificial Intelligence internship project**
(task: *Music Generation with AI*). It is a learning project.

**Objective.** Learn musical patterns from MIDI files and generate new note sequences, then play and download them.

**Method.**
1. MIDI files are loaded and validated (PrettyMIDI); non-drum notes are merged into a monophonic *skyline* melody.
2. Each note becomes a token `(pitch, duration bucket)` and is mapped to an integer id.
3. Sliding windows of N tokens form training pairs: *previous notes → next note*.
4. An LSTM (Embedding → LSTM → Dropout → LSTM → Dense → Softmax) is trained with cross-entropy loss; the best-validation checkpoint is saved.
5. Generation seeds the model with a short real passage from the dataset, then repeatedly samples the next note with **temperature sampling**.
6. Notes are written to MIDI and rendered to WAV (FluidSynth if available, otherwise a simple built-in synthesiser).

**Technologies.** Python, PyTorch, PrettyMIDI, NumPy, Pandas, Streamlit, Plotly, SQLite.

**Honest limitations.** Monophonic only; rests/velocity are not modelled; genre and mood are saved as tags and do **not** steer the model;
quality depends on dataset size and training time.

**Future work.** Polyphony, genre/mood conditioning (conditional LSTM or separate models), Transformer/GAN models, better soundfonts.
""")
