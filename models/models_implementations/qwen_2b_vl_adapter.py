from models.model_interface import OCRModel, OCRInput, OCROutput
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2bAdapter(OCRModel):

    def __init__(self):
        self.id = "qwen2-2b-vl"
        from huggingface_hub import logging as hf_logging
        from transformers.utils import logging as t_logging
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        self.prompt = "You are an OCR engine. Transcribe the Danish text. Return ONLY the text from the image, Do not include coordinates, bounding boxes, labels, or explanations. Output text in a single line"

    def __call__(self,inputs : OCRInput) -> OCROutput:
        img = inputs.image.convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompt}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Transcribe the text on the image. Output text only in a single line."}]},
        ]

        #preprocessing the input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        model_input = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        model_input = model_input.to("cuda")

        #Model inference
        generated_ids = self.model.generate(**model_input, do_sample=False, temperature=0.0, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_input.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return OCROutput(text=output_text)




