# Test set

Target: 300-500 answer pages. Raw images go in `data/raw/` (not committed; student data).
One label file per page in `data/labels/<page_id>.json`.

## Coverage to collect (aim for a mix, not all clean)
- clean flatbed scans
- phone photos (with and without shadows)
- ruled and plain paper
- pages with diagrams
- pages with math / formulas
- messy handwriting
- pages with a header block (USN, name, semester)

## Label file format
```json
{
  "page_id": "p0001",
  "image": "data/raw/p0001.jpg",
  "quality": {"is_ocr_ready": true, "blur": false, "shadow": true, "low_contrast": false, "skew_deg": 2.0},
  "regions": [
    {"bbox": [x1, y1, x2, y2], "type": "usn", "text": "1AB22CS001"},
    {"bbox": [x1, y1, x2, y2], "type": "answer_text", "text": "...", "parent_question": "Q1"}
  ]
}
```
Region types are listed in `src/interface.py` (`REGION_TYPES`).
Shadows and ruled lines are NOT regions: anything detected there counts as a false positive.

## Labelling rules
- Transcribe exactly what the student wrote, including spelling mistakes. No correction.
- Two people label 10% of pages to measure agreement.
- Keep a frozen split once labelling is done: do not edit after Phase 3 starts.

## Needed alongside
- `data/roster.csv`: sample student roster (USN, name, semester) for the header-matching test.
  Use fake or anonymised data if real records can't leave the college.
