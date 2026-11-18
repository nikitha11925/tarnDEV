# Licence table (checked 2026-11-10)

Source: Hugging Face model-card API and GitHub repo API. "Verify" means the tool returned
nothing clear and the licence file must be read by hand. Code licence is not always the same
as weight licence: check both before commercial use.

## Models (weights)
| Model | Licence | Status |
| --- | --- | --- |
| Qwen3-VL (2B, 4B, 8B Instruct) | Apache-2.0 | OK |
| Qwen2.5-VL-7B-Instruct | Apache-2.0 | OK (older) |
| PaddleOCR-VL | Apache-2.0 | OK |
| GLM-OCR | MIT | OK |
| InternVL3-8B / InternVL3.5-8B | Apache-2.0 | OK |
| GOT-OCR2.0 (stepfun-ai) | Apache-2.0 | OK |
| olmOCR-2-7B | Apache-2.0 | OK |
| LightOnOCR-1B | Apache-2.0 | OK |
| TrOCR (base/large handwritten) | MIT | OK (large card shows none: code repo is MIT) |
| PP-DocLayout_plus-L | Apache-2.0 | OK |
| MinerU2.5 | **AGPL-3.0** | Blocked unless legal accepts AGPL (blogs said Apache: they were wrong) |
| Chandra OCR | OpenRAIL | Verify: has use restrictions |
| dots.ocr | unknown | Verify: repo id not found on Hugging Face |

## Code / tools
| Tool | Licence | Status |
| --- | --- | --- |
| PaddleOCR | Apache-2.0 | OK |
| LayoutParser | Apache-2.0 | OK |
| UVDoc | MIT | OK |
| Inv3D (dewarp) | MIT | OK |
| UniMERNet | Apache-2.0 | OK |
| TexTeller | Apache-2.0 | OK |
| Pix2Tex (LaTeX-OCR) | MIT | OK |
| MANIQA | Apache-2.0 | OK |
| DocLayout-YOLO | **AGPL-3.0** | Blocked unless legal accepts AGPL |
| DocTr | not stated (NOASSERTION) | Verify |
| DocEnTr | not stated (NOASSERTION) | Verify |
| PyIQA (IQA-PyTorch) | not stated (NOASSERTION) | Verify: read the repo's LICENSE file by hand |
| DE-GAN | repo not found | Verify the right repo |
| DocIQ | no code found | Verify |

## Hardware note
This machine has an RTX 4050 Laptop GPU with 6 GB VRAM. An 8B VLM will not fit in fp16.
Plan for Qwen3-VL-2B/4B (4-bit for 4B) locally, and use a bigger GPU for the 7-8B runs.
