"""Candidate readers. Each returns (text, seconds, peak_vram_mb). Same prompt for every VLM."""
import gc, os, time
import torch
from PIL import Image

os.environ.setdefault("HF_HOME", r"N:\tarnDEV\model_evaluation\.cache")

PROMPT = ("Transcribe all handwritten text on this page exactly as written, line by line. "
          "Do not correct spelling or grammar. Do not add, summarise or explain anything.")
MAX_PIXELS = 1600 * 1600     # cap input size so latency / VRAM are comparable across models


def load_image(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if im.width * im.height > MAX_PIXELS:
        s = (MAX_PIXELS / (im.width * im.height)) ** 0.5
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return im


class Reader:
    name = "base"

    def load(self): ...
    def _run(self, image: Image.Image) -> str: ...

    def unload(self):
        for a in ("model", "processor"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect(); torch.cuda.empty_cache()

    def read(self, image_path: str):
        im = load_image(image_path)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        text = self._run(im)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        return text, dt, torch.cuda.max_memory_allocated() / 2**20


class Qwen3VL(Reader):
    def __init__(self, size="4B", four_bit=True):
        self.repo = f"Qwen/Qwen3-VL-{size}-Instruct"
        self.four_bit = four_bit
        self.name = f"qwen3-vl-{size.lower()}" + ("-4bit" if four_bit else "")

    def load(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
        kw = dict(device_map="cuda")
        if self.four_bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        else:
            kw["dtype"] = torch.bfloat16
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.repo, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(self.repo)

    def _run(self, image):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": getattr(self, "prompt", PROMPT)}]}]
        inp = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=getattr(self, "max_new", 1500), do_sample=False)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class GLMOCR(Reader):
    name = "glm-ocr"
    repo = "zai-org/GLM-OCR"

    def load(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(self.repo)
        self.model = AutoModelForImageTextToText.from_pretrained(self.repo, dtype=torch.bfloat16, device_map="cuda").eval()

    def _run(self, image):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Text Recognition:"}]}]
        inp = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors="pt").to("cuda")
        inp.pop("token_type_ids", None)
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=getattr(self, "max_new", 1500), do_sample=False)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class PaddleOCRVL(Reader):
    """Element-level recognition with the 0.9B core model (official page parser needs PaddlePaddle)."""
    name = "paddleocr-vl-0.9b"
    repo = "PaddlePaddle/PaddleOCR-VL"

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.model = AutoModelForCausalLM.from_pretrained(self.repo, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(self.repo, trust_remote_code=True)

    def _run(self, image):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "OCR:"}]}]
        inp = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=getattr(self, "max_new", 1500), do_sample=False, max_time=90)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class TrOCR(Reader):
    """Line-level handwriting model (MIT). Pages are cut into lines first (rough projection splitter);
    images that already are a single line (very wide and short) are read directly."""
    def __init__(self, repo="microsoft/trocr-base-handwritten", name="trocr-base"):
        self.repo, self.name = repo, name

    def load(self):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.processor = TrOCRProcessor.from_pretrained(self.repo)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.repo).to("cuda").eval()

    @staticmethod
    def _lines(im):
        import cv2, numpy as np
        g = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        if im.width / im.height > 3:                       # already one line
            return [im]
        ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15)
        rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (im.width // 4, 1)))
        ink = cv2.subtract(ink, cv2.dilate(rules, np.ones((3, 1), np.uint8)))      # drop ruled lines
        prof = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))).sum(axis=1) / 255
        on = prof > max(3, 0.01 * im.width)
        out, start = [], None
        for y, v in enumerate(list(on) + [False]):
            if v and start is None: start = y
            if not v and start is not None:
                if y - start > im.height * 0.012: out.append((max(0, start - 8), min(im.height, y + 8)))
                start = None
        return [im.crop((0, a, im.width, b)) for a, b in out] or [im]

    def _run(self, image):
        lines = self._lines(image)
        texts = []
        for ln in lines:
            pv = self.processor(images=ln, return_tensors="pt").pixel_values.to("cuda")
            with torch.inference_mode():
                ids = self.model.generate(pv, max_new_tokens=96)
            texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0])
        return "\n".join(texts)


READERS = {
    "qwen3vl-2b": lambda: Qwen3VL("2B", four_bit=False),
    "qwen3vl-4b": lambda: Qwen3VL("4B", four_bit=True),
    "glm-ocr": lambda: GLMOCR(),
    "paddleocr-vl": lambda: PaddleOCRVL(),
    "trocr-base": lambda: TrOCR(),
}


# Improved variants: clean page -> split into lines -> normalise -> verbatim prompt, one line at a time.
def _lines(inner_factory, name):
    def make():
        from lines import LineReader
        inner = inner_factory()
        inner.max_new = 120                      # one line never needs more
        return LineReader(inner, name)
    return make

READERS["qwen3vl-2b+lines"] = _lines(READERS["qwen3vl-2b"], "qwen3vl-2b+lines")
READERS["qwen3vl-4b+lines"] = _lines(READERS["qwen3vl-4b"], "qwen3vl-4b+lines")
READERS["glm-ocr+lines"] = _lines(READERS["glm-ocr"], "glm-ocr+lines")
READERS["paddleocr-vl+lines"] = _lines(READERS["paddleocr-vl"], "paddleocr-vl+lines")
