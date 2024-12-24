import torch
from transformers import AutoModel, AutoTokenizer


class HuggingFaceEmbedder:
    def __init__(
        self,
        model_name,
        device=None,
        pooling_method=None,
        max_length=512,
        use_fast_tokenizer=True,
    ):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pooling_method = pooling_method
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=use_fast_tokenizer
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _pool_cls(self, outputs):
        return outputs.last_hidden_state[:, 0, :]

    def _pool_mean(self, outputs, inputs):
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _pool_max(self, outputs, inputs):
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        token_embeddings[
            input_mask_expanded == 0
        ] = -1e9  # Set padding tokens to large negative value
        embeddings, _ = torch.max(token_embeddings, dim=1)
        return embeddings

    def _pool_pooler_output(self, outputs):
        return outputs.pooler_output

    def generate_embeddings(self, sentences, batch_size=32):
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings_list = []

        # instead of batch, split the sentences into chunks of chunk_size
        # ["Lorem ispum dolor sit amet consectetur adipiscing elit"]
        # -> ["Lorem ispum dolor sit amet", "consectetur adipiscing elit"]
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            if self.pooling_method == "cls":
                embeddings = self._pool_cls(outputs)
            elif self.pooling_method == "mean":
                embeddings = self._pool_mean(outputs, inputs)
            elif self.pooling_method == "max":
                embeddings = self._pool_max(outputs, inputs)
            elif self.pooling_method == "pooler_output" and hasattr(
                outputs, "pooler_output"
            ):
                embeddings = self._pool_pooler_output(outputs)
            else:
                # Return raw model output, extracting tensors to add to list
                embeddings_list.append({key: val.cpu() for key, val in outputs.items()})
                continue

            embeddings_list.append(embeddings.cpu())

        if self.pooling_method:
            embeddings = torch.cat(embeddings_list, dim=0)
            return embeddings.numpy()
        else:
            return embeddings_list
