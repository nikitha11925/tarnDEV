# OCR quality metrics, by priority

Scope: quality of OCR output for handwritten answer pages. Each metric says what it protects
the business from. "Gate" means a model that fails it is out regardless of other scores.

## Priority 1: decides the model (gates)
| Metric | Why it matters | How measured |
| --- | --- | --- |
| CER (character error rate) | Core accuracy. "95%" means CER <= 5%. | jiwer on normalised text |
| WER (word error rate) | Grading reads words, not characters. | jiwer |
| Insertion rate (invented text) | Models that "autocorrect" or hallucinate hide what the student wrote, and unfairly change marks. | insertions / reference chars |
| Licence + self-hostable | Must be free, open weights, usable for 5-10 years, no data leaving the college. | docs/licences.md, GATE |
| Fits the hardware | A model that cannot run on the available GPU cannot be used. | peak VRAM, GATE |

## Priority 2: decides between close models
| Metric | Why it matters | How measured |
| --- | --- | --- |
| Latency (s/page) | Throughput for a full exam batch. | wall time per page, median and p95, after warm-up |
| Peak VRAM / RAM | Sets the hardware cost. | torch.cuda.max_memory_allocated |
| Digit and ID accuracy | USN, marks and numbers are the high-risk fields; LLM-style readers are weak at digits. | CER on digit-only spans |
| Consistency (run-to-run) | Same page must give the same text; sampling makes graders disagree. | repeat 3 runs, CER between runs |
| Failure rate | Crashes, empty output, runaway repeated text. | count per 100 pages |
| Script coverage | English, plus Hindi / Kannada if required. | CER per script |

## Priority 3: needed for safe deployment, not for choosing the first model
| Metric | Why it matters | How measured |
| --- | --- | --- |
| Confidence calibration (AUROC, ECE) | Needed to route uncertain pages to a human. | needs per-token scores |
| Inter-model agreement | Free error signal with no ground truth. | CER between two readers |
| Reading order / region accuracy | Visual completeness, later phase. | layout metrics |
| Robustness by condition | Shadow, blur, low resolution. | CER split by page profile |
| Maintenance health | 5-10 year risk: release cadence, community size. | manual check |
| Cost per 1,000 pages | Total cost of ownership. | GPU hours x rate |

## Rules for fair comparison
- Same pages, same GPU, same prompt for every VLM.
- Normalise text the same way for all (case, whitespace, punctuation rule written down).
- Measure latency after one warm-up page; report median and p95.
- Report CER per page, not only the mean: a few terrible pages matter more than the average.
- Prompt for VLMs: "Transcribe exactly. Do not correct spelling. Do not add text."
