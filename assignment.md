# Assignment: Implementing DAAM for CapSpeech

---

### Objective

Adapt the DAAM cross-attention attribution pipeline to CapSpeech's TTS model to visualize how text tokens correlate with regions in generated speech (e.g., time-aligned spectrogram regions).

---

### Tasks

#### 1. Attention Extraction

- **Instrument the CapSpeech model** to capture cross-attention maps during inference.
- **Collect attention tensors** for each text token across layers and diffusion steps (if applicable).

#### 2. Mapping to Speech

- **Define a mapping** from attention positions to spectrogram time (and optionally frequency) bins.
- **Upsample attention maps** to align with the spectrogram time axis.

#### 3. Aggregation

- **Aggregate attention** across layers and steps into per-token heatmaps over the speech representation.
- **Normalize maps** so they are comparable across tokens.

#### 4. Visualization

- **Overlay token heatmaps** on spectrogram plots.
- **Produce side-by-side outputs** showing text tokens and corresponding attention regions.

---

### Deliverables

- **Code module** (e.g., `daam_capspeech.py`) containing:
  - `extract_attn()`
  - `upsample_attn()`
  - `aggregate_attn()`
  - `visualize_maps()`
- **Five example visualizations** (token -> spectrogram overlays).
- **A brief report** (1–2 pages) summarizing implementation choices and sample outputs.

---

### References

- Tang et al., *What the DAAM: Interpreting Stable Diffusion Using Cross Attention* (ACL / arXiv).
  - Github: [https://github.com/castorini/daam](https://github.com/castorini/daam)
- Wang et al., *CapSpeech: Enabling Downstream Applications in Style-Captioned Text-to-Speech* (arXiv).
  - Github: [https://github.com/WangHelin1997/CapSpeech](https://github.com/WangHelin1997/CapSpeech)

