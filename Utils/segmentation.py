from config import TARGET_SR, WINDOW_SEC, HOP_SEC

def sliding_windows(y):
    win_len = int(TARGET_SR * WINDOW_SEC)
    hop_len = int(TARGET_SR * HOP_SEC)
    for start in range(0, len(y) - win_len + 1, hop_len):
        yield y[start:start + win_len], start / TARGET_SR