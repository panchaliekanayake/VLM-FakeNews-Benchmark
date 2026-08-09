"""LLaVA-v1.6 Vicuna-7B adapter using the official LLaVA package."""

from pathlib import Path

from .base import ModelAdapter


class LlavaAdapter(ModelAdapter):
    def __init__(self, model_path: str, gpu_id: int = 0, revision: str | None = None, **_: object) -> None:
        import torch
        if revision and "/" in model_path and not Path(model_path).exists():
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(repo_id=model_path, revision=revision)
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        disable_torch_init()
        self.torch = torch
        name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(model_path, None, name)
        lowered = name.lower()
        self.conversation_mode = (
            "llava_llama_2" if "llama-2" in lowered else
            "mistral_instruct" if "mistral" in lowered else
            "chatml_direct" if "v1.6-34b" in lowered else
            "llava_v1" if "v1" in lowered else
            "mpt" if "mpt" in lowered else "llava_v0"
        )

    def generate(self, image_path: Path, prompt: str) -> str:
        from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image_sizes = [image.size]
        images = process_images([image], self.processor, self.model.config).to(
            self.model.device, dtype=self.torch.float16
        )
        token = DEFAULT_IMAGE_TOKEN
        if self.model.config.mm_use_im_start_end:
            token = DEFAULT_IM_START_TOKEN + token + DEFAULT_IM_END_TOKEN
        conversation = conv_templates[self.conversation_mode].copy()
        conversation.append_message(conversation.roles[0], f"{token}\n{prompt}")
        conversation.append_message(conversation.roles[1], None)
        input_ids = tokenizer_image_token(
            conversation.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids, images=images, image_sizes=image_sizes, do_sample=False,
                temperature=0.0, num_beams=1, max_new_tokens=512, use_cache=True,
            )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
