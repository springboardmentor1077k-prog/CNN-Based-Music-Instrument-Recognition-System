"""
InstruNet - Utility Modules

This package contains utility functions for audio processing,
feature extraction, visualization, and report generation.
"""

from .aggregation import aggregate, moving_average
from .audio import load_audio, stereo_to_mono, peak_normalize, trim_silence, fix_duration
from .features import generate_log_mel, fix_mel_frames
from .segmentation import sliding_windows
from .visualization import plot_intensity
from .io import intensity_to_json
from .pdf_report import generate_pdf_report

__all__ = [
    'aggregate',
    'moving_average',
    'load_audio',
    'stereo_to_mono',
    'peak_normalize',
    'trim_silence',
    'fix_duration',
    'generate_log_mel',
    'fix_mel_frames',
    'sliding_windows',
    'plot_intensity',
    'intensity_to_json',
    'generate_pdf_report',
]