# TODO

- [x] explore the audio.
- [x] load the audio in python (librosa).
- [x] identify stereo
 	- [x] convert to mono
- [x] convert to 16khz sampling rate.
- [x] generate the waveform graph (matplotlib).

## Notes
- **Dataset Exploration:**
  - Total files: 6705
  - Classes found: cel, cla, flu, gac, gel, org, pia, sax, tru, vio, voi.
  - Audio Format: 44.1kHz, Stereo (2 channels), ~3s duration.
- **Processing:**
  - Successfully implemented loading, mono conversion, and resampling to 16kHz in `src/audio_processor.py`.
  - Sample waveform generated.
