from transformers import AutoTokenizer, AutoModel
import torch

class HuggingFaceEmbedder:
    def __init__(self, model_name, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def generate_embeddings(self, sentences):
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state
        return embeddings.cpu().numpy()