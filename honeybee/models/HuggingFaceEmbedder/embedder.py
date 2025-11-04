import logging
from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# Custom exceptions for better error handling
class ModelAccessError(Exception):
    """Raised when model requires authentication or access approval"""

    pass


class ModelNotFoundError(Exception):
    """Raised when model doesn't exist on HuggingFace Hub"""

    pass


class HuggingFaceEmbedder:
    def __init__(
        self,
        model_name,
        device=None,
        pooling_method=None,
        max_length=512,
        use_fast_tokenizer=True,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.model_name = model_name

        try:
            # Try to load tokenizer
            logger.info(f"Loading tokenizer for model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=use_fast_tokenizer
            )

            # Try to load model
            logger.info(f"Loading model: {model_name}")
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded model {model_name} on {self.device}")

        except OSError as e:
            # Handle authentication/access errors (401, 403)
            error_msg = str(e).lower()
            if "401" in error_msg or "403" in error_msg or "gated" in error_msg:
                self._raise_access_error(model_name, e)
            # Handle model not found (404)
            elif (
                "404" in error_msg
                or "not find" in error_msg
                or "does not appear" in error_msg
                or "repository not found" in error_msg
            ):
                self._raise_not_found_error(model_name, e)
            else:
                # Other OS errors (network, disk, etc.)
                raise RuntimeError(
                    f"Failed to load model '{model_name}': {e}\n\n"
                    f"This might be due to:\n"
                    f"  • Network connectivity issues\n"
                    f"  • Insufficient disk space\n"
                    f"  • Corrupted cache\n\n"
                    f"Try: huggingface-cli delete-cache"
                ) from e

        except Exception as e:
            # Catch-all for other errors
            raise RuntimeError(
                f"Unexpected error loading model '{model_name}': {type(e).__name__}: {e}\n\n"
                f"Suggested models that don't require authentication:\n"
                + self._format_model_suggestions()
            ) from e

    def _raise_access_error(self, model_name: str, original_error: Exception):
        """Raise informative error for gated/authentication-required models"""
        message = (
            f"\n{'='*70}\n"
            f"MODEL ACCESS ERROR\n"
            f"{'='*70}\n\n"
            f"Cannot access model: '{model_name}'\n\n"
            f"This model requires authentication or access approval.\n\n"
            f"To use this model:\n\n"
            f"1. Create a HuggingFace account:\n"
            f"   https://huggingface.co/join\n\n"
            f"2. Request access on the model page:\n"
            f"   https://huggingface.co/{model_name}\n"
            f"   (Click 'Request access' and wait for approval)\n\n"
            f"3. After approval, authenticate:\n"
            f"   huggingface-cli login\n"
            f"   (Or set HF_TOKEN environment variable)\n\n"
            f"{'='*70}\n"
            f"ALTERNATIVE: Use these open-access models instead:\n"
            f"{'='*70}\n"
            + self._format_model_suggestions()
            + f"\n{'='*70}\n"
        )
        raise ModelAccessError(message) from original_error

    def _raise_not_found_error(self, model_name: str, original_error: Exception):
        """Raise informative error for non-existent models"""
        message = (
            f"\n{'='*70}\n"
            f"MODEL NOT FOUND\n"
            f"{'='*70}\n\n"
            f"Model '{model_name}' does not exist on HuggingFace Hub.\n\n"
            f"Please check:\n"
            f"  • Model name spelling\n"
            f"  • Model is public (not private)\n"
            f"  • Model URL: https://huggingface.co/{model_name}\n\n"
            f"Search for models at: https://huggingface.co/models\n\n"
            f"{'='*70}\n"
            f"Try these verified biomedical models:\n"
            f"{'='*70}\n"
            + self._format_model_suggestions()
            + f"\n{'='*70}\n"
        )
        raise ModelNotFoundError(message) from original_error

    @staticmethod
    def _format_model_suggestions() -> str:
        """Format open-access model suggestions"""
        suggestions = HuggingFaceEmbedder.get_recommended_open_models()
        lines = []

        lines.append("\nBiomedical Models (no authentication required):")
        for model in suggestions["biomedical"]:
            lines.append(f"  • {model['name']}")
            lines.append(f"    {model['description']}")

        lines.append("\nGeneral-Purpose Models (lightweight):")
        for model in suggestions["general"]:
            lines.append(f"  • {model['name']}")
            lines.append(f"    {model['description']}")

        return "\n".join(lines)

    @staticmethod
    def get_recommended_open_models() -> Dict[str, List[Dict[str, str]]]:
        """
        Get list of recommended open-access models that don't require authentication

        Returns:
            Dictionary with 'biomedical' and 'general' model lists
        """
        return {
            "biomedical": [
                {
                    "name": "emilyalsentzer/Bio_ClinicalBERT",
                    "description": "Clinical BERT trained on MIMIC notes (110M params)",
                },
                {
                    "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "description": "PubMedBERT trained on PubMed abstracts (110M params)",
                },
                {
                    "name": "dmis-lab/biobert-v1.1",
                    "description": "BioBERT for biomedical text mining (110M params)",
                },
                {
                    "name": "allenai/scibert_scivocab_uncased",
                    "description": "SciBERT for scientific publications (110M params)",
                },
            ],
            "general": [
                {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "description": "Fast sentence embeddings (23M params)",
                },
                {
                    "name": "bert-base-uncased",
                    "description": "Original BERT base model (110M params)",
                },
                {
                    "name": "roberta-base",
                    "description": "RoBERTa base model (125M params)",
                },
            ],
        }

    def _pool_cls(self, outputs):
        return outputs.last_hidden_state[:, 0, :]

    def _pool_mean(self, outputs, inputs):
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _pool_max(self, outputs, inputs):
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
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
            elif self.pooling_method == "pooler_output" and hasattr(outputs, "pooler_output"):
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
