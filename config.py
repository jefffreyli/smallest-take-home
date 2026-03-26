"""Runtime configuration for daam_capspeech.py.
"""

TASK = "CapTTS"  # PT | CapTTS | EmoCapTTS | AccCapTTS | AgentTTS
OUTPUT_DIR = "outputs"
DEVICE = "auto"  # "auto" picks cuda:0 when available else cpu
STEPS = 25
CFG = 2.0
MAX_TOKENS = 20
SEED = 42
