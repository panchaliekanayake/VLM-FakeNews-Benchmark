"""OTTER-Image-MPT-7B adapter."""

from pathlib import Path

from .base import ModelAdapter


class OtterAdapter(ModelAdapter):
    def __init__(self, model_path: str, gpu_id: int = 0, revision: str | None = None, **_: object) -> None:
        import torch
        from transformers import CLIPImageProcessor
        try:
            from otter_ai import OtterForConditionalGeneration
        except ImportError as exc:
            raise ImportError("Install the official OTTER package before using this adapter.") from exc

        self.torch = torch
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = (
            OtterForConditionalGeneration.from_pretrained(model_path, revision=revision)
            .to(device=self.device, dtype=dtype)
            .eval()
        )
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()

    def generate(self, image_path: Path, prompt: str) -> str:
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB"))
        vision = self.image_processor.preprocess(
            [image_array], return_tensors="pt", input_data_format="channels_last"
        )["pixel_values"].unsqueeze(1).unsqueeze(0).to(device=self.device, dtype=self.model.dtype)
        language = self.tokenizer([f"<image>User: {prompt} GPT:<answer>"], return_tensors="pt", padding=True)
        language = {key: value.to(self.device) for key, value in language.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                vision_x=vision, lang_x=language["input_ids"], attention_mask=language["attention_mask"],
                max_new_tokens=64, num_beams=3, no_repeat_ngram_size=3,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).rsplit("GPT:<answer>", 1)[-1].strip()
