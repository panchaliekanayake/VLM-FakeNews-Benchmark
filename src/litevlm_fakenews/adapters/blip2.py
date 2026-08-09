"""BLIP-2 Flan-T5-XL adapter."""

from pathlib import Path

from .base import ModelAdapter


class Blip2Adapter(ModelAdapter):
    def __init__(self, model_path: str, gpu_id: int = 0, revision: str | None = None, **_: object) -> None:
        import torch
        from transformers import AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor, BlipImageProcessor

        self.torch = torch
        image_processor = BlipImageProcessor.from_pretrained(model_path, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, revision=revision)
        self.processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = (
            Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, revision=revision)
            .to(self.device)
            .eval()
        )

    def generate(self, image_path: Path, prompt: str) -> str:
        from PIL import Image, ImageOps

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=32, num_beams=1)
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()
