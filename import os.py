import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG =========
DATASET_ROOT = r"C:\Users\ADMIN\Downloads\music_dataset"  # your dataset root
SAVE_PROCESSED_FILES = False   # True = save *_processed.wav files
TARGET_SR = 60000              # 60 kHz

# How many example files per instrument to CONSIDER for plotting
# (we'll usually just take the first one)
MAX_FILES_PER_INSTRUMENT_FOR_PLOT = 1
# ==========================


def process_audio_file(file_path, save_file=False):
    # Load with original sample rate and channels
    y, sr = librosa.load(file_path, sr=None, mono=False)

    # Mono / stereo
    if y.ndim == 1:
        is_stereo = False
        y_mono = y
        num_samples = len(y)
    else:
        is_stereo = True
        y_mono = librosa.to_mono(y)
        num_samples = y.shape[1]

    sr_old = sr

    # Resample only if sr < TARGET_SR
    if sr_old < TARGET_SR:
        y_processed = librosa.resample(y_mono, orig_sr=sr_old, target_sr=TARGET_SR)
        sr_new = TARGET_SR
        resampled = True
    else:
        y_processed = y_mono
        sr_new = sr_old
        resampled = False

    # Optional save (separate processed file)
    if save_file:
        base, ext = os.path.splitext(file_path)
        new_path = base + "_processed" + ext
        sf.write(new_path, y_processed, sr_new)
        print(f"  Saved processed file → {new_path}")

    return y_processed, sr_old, sr_new, is_stereo, resampled, num_samples


def plot_all_instruments(waveforms, sample_rates, instrument_names, file_names):
    """
    One big template:
    - each instrument = one subplot
    - 1 waveform per instrument
    """
    n = len(waveforms)
    if n == 0:
        print("No waveforms collected for plotting.")
        return

    plt.figure(figsize=(10, 2.5 * n))
    plt.suptitle("Waveforms (one per instrument)", fontsize=14, y=0.98)

    for i, (y, sr, inst, fname) in enumerate(
        zip(waveforms, sample_rates, instrument_names, file_names), start=1
    ):
        duration = len(y) / sr
        t = np.linspace(0, duration, num=len(y))

        ax = plt.subplot(n, 1, i)
        ax.plot(t, y)
        ax.set_ylabel("Amp")
        ax.set_title(f"{inst} - {fname}", fontsize=9)

        if i == n:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])  # hide x labels for middle plots

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def explore_dataset(root_folder, save_files=False):
    print(f"Scanning root folder: {root_folder}\n")

    total_files = 0
    total_resampled = 0
    total_stereo = 0
    total_mono = 0

    # These store ONE example waveform per instrument for the final template
    waveforms_all = []
    sample_rates_all = []
    instrument_names_all = []
    file_names_all = []

    # Each subfolder here is an instrument (Accordion, Violin, etc.)
    for instrument_name in sorted(os.listdir(root_folder)):
        instrument_path = os.path.join(root_folder, instrument_name)

        if not os.path.isdir(instrument_path):
            continue  # skip non-folders

        print(f"\n================ Instrument Folder: {instrument_name} ================")

        folder_files = 0
        folder_resampled = 0
        folder_stereo = 0
        folder_mono = 0

        example_taken_for_this_instrument = 0

        for filename in sorted(os.listdir(instrument_path)):
            if not filename.lower().endswith(".wav"):
                continue

            filepath = os.path.join(instrument_path, filename)
            folder_files += 1
            total_files += 1

            print(f"\nFile: {filepath}")

            try:
                (
                    y_proc,
                    sr_old,
                    sr_new,
                    is_stereo,
                    resampled,
                    num_samples,
                ) = process_audio_file(filepath, save_file=save_files)
            except Exception as e:
                print(f"  Error processing file: {e}")
                continue

            duration = num_samples / sr_old

            if is_stereo:
                folder_stereo += 1
                total_stereo += 1
            else:
                folder_mono += 1
                total_mono += 1

            if resampled:
                folder_resampled += 1
                total_resampled += 1

            print("  Channels:        ", "STEREO" if is_stereo else "MONO")
            print(f"  Original SR:      {sr_old} Hz")
            print(f"  Samples (orig):   {num_samples}")
            print(f"  Duration (orig):  {duration:.2f} sec")
            print(f"  New SR:           {sr_new} Hz")
            print("  Resampled:       ", "Yes" if resampled else "No")

            # Store just ONE example (or up to MAX_FILES_PER_INSTRUMENT_FOR_PLOT)
            if example_taken_for_this_instrument < MAX_FILES_PER_INSTRUMENT_FOR_PLOT:
                waveforms_all.append(y_proc)
                sample_rates_all.append(sr_new)
                instrument_names_all.append(instrument_name)
                file_names_all.append(filename)
                example_taken_for_this_instrument += 1

        # Per-folder summary
        print(f"\n--- Summary for {instrument_name} ---")
        print(f"  Files in folder:   {folder_files}")
        print(f"  Mono:              {folder_mono}")
        print(f"  Stereo:            {folder_stereo}")
        print(f"  Resampled:         {folder_resampled}")
        print("------------------------------------")

    # Global summary
    print("\n========== TOTAL SUMMARY ==========")
    print(f"Total .wav files:   {total_files}")
    print(f"Total MONO:         {total_mono}")
    print(f"Total STEREO:       {total_stereo}")
    print(f"Total resampled:    {total_resampled}")
    print("===================================")

    # Now, AFTER scanning ALL folders, plot ONE template with all instruments
    print("\nCreating global waveform template (one per instrument)...")
    plot_all_instruments(waveforms_all, sample_rates_all, instrument_names_all, file_names_all)


if __name__ == "__main__":
    explore_dataset(
        DATASET_ROOT,
        save_files=SAVE_PROCESSED_FILES,
    )
