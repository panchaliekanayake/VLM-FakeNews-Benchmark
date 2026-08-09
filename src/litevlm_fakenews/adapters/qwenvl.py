"""Original Qwen-VL adapter."""

from pathlib import Path

from .base import ModelAdapter


class QwenVLAdapter(ModelAdapter):
    def __init__(self, model_path: str, gpu_id: int = 0, revision: str | None = None, **_: object) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision=revision)
        device_map: object = {"": gpu_id} if torch.cuda.is_available() else "cpu"
        kwargs = {"trust_remote_code": True, "device_map": device_map, "revision": revision}
        if torch.cuda.is_available():
            kwargs["fp16"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()

    def generate(self, image_path: Path, prompt: str) -> str:
        query = self.tokenizer.from_list_format([{"image": str(image_path)}, {"text": prompt}])
        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=128)
        decoded = self.tokenizer.decode(output[0].cpu(), skip_special_tokens=False)
        if "</img>" in decoded:
            decoded = decoded.split("</img>", 1)[1]
        if "<|endoftext|>" in decoded:
            decoded = decoded.split("<|endoftext|>", 1)[0]
        return decoded.strip()
