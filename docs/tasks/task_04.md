## Complete preprocessing pipeline
- **Pipeline :** 
	- two audio files: 1. 44.1khz sterio variable length and 2. 16khz mono 3 seconds
	- We need the pipeline so that the cnn can generalize the spectrogram input.
	- The model will become high in accuracy and consistently accurate.
1. Load the audio - extract the metadata to know the data (librosa loads audios as float_32).
2. Convert to mono - converting stereo to mono (remove the left and right channels and merge into one)
3. Resampling to 16khz - for faster training (16khz is good enough).
4. Normalizing the amplitude - uniform dynamic range.
5. Silence trimming - Remove the empty or nearly silent parts of the audios.
6. Padding the duration - Duration fixing
7. Export the clean audio
- from next week the cnn model depends on the preprocessing.
