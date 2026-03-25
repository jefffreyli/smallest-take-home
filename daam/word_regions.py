"""Map transcript words to spectrogram time regions via phoneme counts."""

from __future__ import annotations

from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CapSpeech"))


def words_to_time_regions(
    transcript: str,
    text_tokenizer,
    T_spec: int,
) -> List[Tuple[str, int, int]]:
    """Map each transcript word to a spectrogram time range.

    Phoneme counts are used to allocate frames proportionally so that
    longer words (more phonemes) receive more time frames.

    Parameters
    ----------
    transcript : str
        The spoken text (e.g. ``"Hello world."``).
    text_tokenizer : g2p_en.G2p
        Grapheme-to-phoneme converter used by CapSpeech.
    T_spec : int
        Total number of spectrogram time frames.

    Returns
    -------
    list of (word, t_start, t_end)
        One entry per word.  ``t_end`` is exclusive.
    """
    from capspeech.nar.generate import valid_symbols

    words = transcript.split()
    if not words:
        return []

    phn_counts: List[int] = []
    for w in words:
        raw = text_tokenizer(w)
        raw = [p.replace(" ", "<BLK>") for p in raw]
        raw = [p for p in raw if p in valid_symbols]
        phn_counts.append(max(len(raw), 1))

    total_phns = sum(phn_counts)
    regions: List[Tuple[str, int, int]] = []
    cursor = 0
    for i, (w, pc) in enumerate(zip(words, phn_counts)):
        if i < len(words) - 1:
            span = max(int(round(T_spec * pc / total_phns)), 1)
        else:
            span = T_spec - cursor
        t_start = cursor
        t_end = min(cursor + span, T_spec)
        regions.append((w, t_start, t_end))
        cursor = t_end

    return regions
