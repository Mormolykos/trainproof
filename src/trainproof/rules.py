"""Deterministic thresholds for verdicts."""

# -----------------
# DATA SUBCOMMAND
# -----------------

# Maximum allowed duration for a single audio file in seconds.
MAX_AUDIO_DURATION_SEC = 25.0

# Minimum allowed duration for a single audio file in seconds.
MIN_AUDIO_DURATION_SEC = 0.5

# Peak amplitude threshold above which we consider audio to be clipping.
MAX_CLIPPING_PEAK = 0.99

# Maximum allowed continuous silence in seconds before warning.
MAX_SILENCE_SEC = 2.0

# Amplitude threshold for considering a frame as "silent" (for silence detection).
SILENCE_AMPLITUDE = 0.005

# Flag text/audio alignment outliers whose chars-per-second rate deviates from
# the corpus median by more than this ratio (in either direction).
CHARS_PER_SEC_OUTLIER_RATIO = 3.0

# -----------------
# TOKENIZER SUBCOMMAND
# -----------------

# Minimum fraction of the corpus vocabulary that must be covered by the tokenizer.
MIN_VOCAB_COVERAGE = 0.999

# Maximum acceptable Out-Of-Vocabulary (OOV) rate.
MAX_OOV_RATE = 0.001

# Maximum expected tokens per second of audio (blowout warning).
MAX_TOKENS_PER_SEC = 50.0

# -----------------
# EPOCH SUBCOMMAND
# -----------------

# If loss increases by more than this ratio from the minimum loss, flag as diverging.
MAX_LOSS_DIVERGENCE_RATIO = 1.5

# Minimum required variation (std/mean) in the loss curve to not be considered "flat/dead".
MIN_LOSS_VARIATION = 0.001

# Maximum allowable spike in gradient norm compared to the median grad norm.
MAX_GRAD_NORM_SPIKE_RATIO = 10.0

# If the learning rate is strictly zero for more than this fraction of the epoch, it's a warning.
MAX_ZERO_LR_FRACTION = 0.1
