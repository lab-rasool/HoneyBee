import os
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import pandas as pd
from typing import Optional
from datasets import Dataset, load_dataset

class TCGASurvivalProcessor:
    """
    A processor for creating survival analysis datasets from TCGA clinical and pathology data.

    This class handles loading, cleaning, merging, and filtering TCGA datasets to create
    a final dataset suitable for survival analysis with both clinical text and pathology reports.

    Attributes:
        clinical_dataset: Hugging Face dataset containing clinical information
        pathology_dataset: Hugging Face dataset containing pathology reports
        merged_df: Pandas DataFrame containing the merged and processed data
        final_dataset: Hugging Face Dataset ready for analysis
    """

    def __init__(self):
        """Initialize the TCGA Survival Processor."""
        self.clinical_dataset = None
        self.pathology_dataset = None
        self.merged_df = None
        self.final_dataset = None

    def load_datasets(self) -> None:
        """
        Load TCGA clinical and pathology report datasets from Hugging Face.

        Removes embedding columns that are not needed for survival analysis.
        """
        print("Loading TCGA datasets...")

        # Load clinical dataset
        self.clinical_dataset = load_dataset(
            "Lab-Rasool/TCGA", "clinical", split="gatortron"
        )
        self.clinical_dataset = self.clinical_dataset.remove_columns(
            ["embedding_shape", "embedding"]
        )

        # Load pathology report dataset
        self.pathology_dataset = load_dataset(
            "Lab-Rasool/TCGA", "pathology_report", split="gatortron"
        )
        self.pathology_dataset = self.pathology_dataset.remove_columns(
            ["embedding_shape", "embedding"]
        )

        print(
            f"Loaded {len(self.clinical_dataset)} clinical records and {len(self.pathology_dataset)} pathology reports"
        )

    def _calculate_survival_time(self, row: pd.Series) -> Optional[float]:
        """
        Calculate survival time based on vital status.

        Args:
            row: A pandas Series containing vital_status, days_to_death, and days_to_last_follow_up

        Returns:
            Survival time in days or NaN if data is missing/invalid
        """
        if pd.isna(row["vital_status"]) or row["vital_status"] == "Not Reported":
            return np.nan
        elif row["vital_status"].lower() == "dead":
            return row["days_to_death"]
        elif row["vital_status"].lower() == "alive":
            return row["days_to_last_follow_up"]
        else:
            return np.nan

    def _calculate_survival_status(self, row: pd.Series) -> Optional[float]:
        """
        Calculate survival status (event indicator).

        Args:
            row: A pandas Series containing vital_status

        Returns:
            1.0 for death (event), 0.0 for alive/censored, or NaN if data is missing/invalid
        """
        if pd.isna(row["vital_status"]) or row["vital_status"] == "Not Reported":
            return np.nan
        elif row["vital_status"].lower() == "dead":
            return 1.0
        elif row["vital_status"].lower() == "alive":
            return 0.0
        else:
            return np.nan

    def process_clinical_data(self) -> pd.DataFrame:
        """
        Process clinical data to extract survival information.

        Returns:
            Processed clinical DataFrame with survival time and status
        """
        # Convert to pandas
        clinical_df = self.clinical_dataset.to_pandas()

        # Select relevant columns
        clinical_subset = clinical_df[
            [
                "project_id",
                "case_submitter_id",
                "text",
                "days_to_death",
                "days_to_last_follow_up",
                "vital_status",
            ]
        ].copy()

        # Convert string columns to numeric
        clinical_subset["days_to_death"] = pd.to_numeric(
            clinical_subset["days_to_death"], errors="coerce"
        )
        clinical_subset["days_to_last_follow_up"] = pd.to_numeric(
            clinical_subset["days_to_last_follow_up"], errors="coerce"
        )

        # Rename text column
        clinical_subset = clinical_subset.rename(columns={"text": "clinical_text"})

        # Calculate survival variables
        clinical_subset["survival_time_days"] = clinical_subset.apply(
            self._calculate_survival_time, axis=1
        )
        clinical_subset["survival_status"] = clinical_subset.apply(
            self._calculate_survival_status, axis=1
        )

        # Drop intermediate columns
        clinical_subset = clinical_subset.drop(
            ["days_to_death", "days_to_last_follow_up", "vital_status"], axis=1
        )

        return clinical_subset

    def process_pathology_data(self) -> pd.DataFrame:
        """
        Process pathology report data.

        Returns:
            Processed pathology DataFrame with renamed columns
        """
        # Convert to pandas
        pathology_df = self.pathology_dataset.to_pandas()

        # Select and rename columns
        pathology_subset = pathology_df[["PatientID", "report_text"]].copy()
        pathology_subset = pathology_subset.rename(
            columns={"report_text": "pathology_text"}
        )

        return pathology_subset

    def merge_and_filter(
        self, clinical_df: pd.DataFrame, pathology_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge clinical and pathology data and filter for valid survival cases.

        Args:
            clinical_df: Processed clinical DataFrame
            pathology_df: Processed pathology DataFrame

        Returns:
            Merged and filtered DataFrame ready for survival analysis
        """
        # Merge datasets
        merged_df = pd.merge(
            clinical_df,
            pathology_df,
            left_on="case_submitter_id",
            right_on="PatientID",
            how="inner",
        )
        merged_df = merged_df.drop("PatientID", axis=1)

        # Filter for valid survival data
        merged_df_complete = merged_df[
            (merged_df["survival_time_days"].notna())
            & (merged_df["survival_status"].notna())
            & (merged_df["survival_time_days"] > 0)
        ].copy()

        return merged_df_complete

    def create_dataset(self) -> Dataset:
        """
        Create the final survival analysis dataset.

        This method orchestrates the entire pipeline: loading, processing, merging, and filtering.

        Returns:
            Hugging Face Dataset ready for survival analysis
        """
        # Load data if not already loaded
        if self.clinical_dataset is None or self.pathology_dataset is None:
            self.load_datasets()

        # Process clinical and pathology data
        clinical_df = self.process_clinical_data()
        pathology_df = self.process_pathology_data()

        # Merge and filter
        self.merged_df = self.merge_and_filter(clinical_df, pathology_df)

        # Convert to Hugging Face Dataset
        self.final_dataset = Dataset.from_pandas(self.merged_df.reset_index(drop=True))

        print(f"Created dataset with {len(self.merged_df)} patients")
        print(
            f"Mortality rate: {(self.merged_df['survival_status'] == 1).mean() * 100:.1f}%"
        )

        return self.final_dataset

    def get_summary_statistics(self) -> dict:
        """
        Get summary statistics for the processed dataset.

        Returns:
            Dictionary containing key statistics about the dataset
        """
        if self.merged_df is None:
            raise ValueError("Dataset not yet created. Run create_dataset() first.")

        stats = {
            "n_patients": len(self.merged_df),
            "n_deaths": (self.merged_df["survival_status"] == 1).sum(),
            "n_censored": (self.merged_df["survival_status"] == 0).sum(),
            "mortality_rate": (self.merged_df["survival_status"] == 1).mean(),
            "median_survival_time": self.merged_df["survival_time_days"].median(),
            "mean_survival_time": self.merged_df["survival_time_days"].mean(),
            "min_survival_time": self.merged_df["survival_time_days"].min(),
            "max_survival_time": self.merged_df["survival_time_days"].max(),
            "n_projects": self.merged_df["project_id"].nunique(),
            "projects": self.merged_df["project_id"].unique().tolist(),
        }

        return stats

    def save_dataset(self, path: str, format: str = "csv") -> None:
        """
        Save the processed dataset.

        Args:
            path: Path to save the dataset
            format: Format to save in ('csv', 'parquet', 'hf_dataset')
        """
        if self.merged_df is None:
            raise ValueError("Dataset not yet created. Run create_dataset() first.")

        if format == "csv":
            self.merged_df.to_csv(path, index=False)
        elif format == "parquet":
            self.merged_df.to_parquet(path, index=False)
        elif format == "hf_dataset":
            self.final_dataset.save_to_disk(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Dataset saved to {path}")

    def get_project_subset(self, project_id: str) -> pd.DataFrame:
        """
        Get subset of data for a specific TCGA project.

        Args:
            project_id: TCGA project identifier (e.g., 'TCGA-BRCA')

        Returns:
            DataFrame containing only data from the specified project
        """
        if self.merged_df is None:
            raise ValueError("Dataset not yet created. Run create_dataset() first.")

        subset = self.merged_df[self.merged_df["project_id"] == project_id].copy()
        return subset

class UnifiedEmbeddingsGenerator:
    def __init__(self, output_dir="/mnt/f/Projects/HoneyBee/results/llm/embeddings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.models_config = {
            'gatortron': {
                'model_name': 'UFNLP/gatortron-base',
                'max_length': 512,
                'batch_size': 8,
                'model_type': 'standard'
            },
            'qwen': {
                'model_name': 'Qwen/Qwen3-Embedding-0.6B',
                'max_length': 512,
                'batch_size': 8,
                'model_type': 'standard'
            },
            'llama': {
                'model_name': 'meta-llama/Llama-3.2-1B',
                'max_length': 2048,
                'batch_size': 4,
                'model_type': 'llama'
            },
            'medgemma': {
                'model_name': 'google/medgemma-4b-pt',
                'max_length': None,
                'batch_size': 1,
                'model_type': 'medgemma'
            }
        }
        
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to transformer outputs"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def load_data(self):
        """Load TCGA dataset"""
        print("Loading TCGA dataset...")
        processor = TCGASurvivalProcessor()
        processor.create_dataset()
        df = processor.merged_df
        
        print(f"Dataset loaded with {len(df)} samples")
        print(f"Cancer types: {df['project_id'].value_counts().to_dict()}")
        
        return df
    
    def check_existing_embeddings(self, model_key, text_type):
        """Check if embeddings already exist"""
        file_path = os.path.join(self.output_dir, f"{text_type}_{model_key}_embeddings.pkl")
        return os.path.exists(file_path)
    
    def generate_standard_embeddings(self, model_name, texts, max_length, batch_size):
        """Generate embeddings using standard transformer approach"""
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        model.eval()
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**encoded)
                batch_embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def generate_llama_embeddings(self, model_name, texts, max_length, batch_size):
        """Generate embeddings using Llama model with special handling"""
        print(f"Loading Llama model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Llama embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**encoded)
                batch_embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def generate_medgemma_embeddings(self, model_name, texts):
        """Generate embeddings using MedGemma model with special handling"""
        print(f"Loading MedGemma model: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        model.eval()
        
        embeddings = []
        checkpoint_file = os.path.join(self.output_dir, "medgemma_checkpoint.pkl")
        
        # Check for checkpoint
        start_idx = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                embeddings = checkpoint_data['embeddings']
                start_idx = len(embeddings)
                print(f"Resuming from checkpoint at index {start_idx}")
        
        for i in tqdm(range(start_idx, len(texts)), desc="Generating MedGemma embeddings"):
            text = texts[i]
            
            try:
                inputs = processor(text=text, return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    embedding = hidden_states.mean(dim=1).squeeze().float().cpu().numpy()
                    embeddings.append(embedding)
                
                # Save checkpoint every 1000 samples
                if (i + 1) % 1000 == 0:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'embeddings': embeddings}, f)
                    print(f"Checkpoint saved at index {i + 1}")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Use zero embedding as fallback
                embeddings.append(np.zeros(4096))
        
        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        return np.array(embeddings)
    
    def save_embeddings(self, embeddings, patient_ids, project_ids, model_key, text_type):
        """Save embeddings to pickle file"""
        output_file = os.path.join(self.output_dir, f"{text_type}_{model_key}_embeddings.pkl")
        
        data = {
            'embeddings': embeddings,
            'patient_ids': patient_ids,
            'project_ids': project_ids
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {text_type} embeddings for {model_key} to {output_file}")
        print(f"Embedding shape: {embeddings.shape}")
    
    def generate_embeddings_for_model(self, model_key, df):
        """Generate embeddings for a specific model"""
        config = self.models_config[model_key]
        
        # Process clinical texts
        if not self.check_existing_embeddings(model_key, 'clinical'):
            print(f"\nGenerating clinical embeddings for {model_key}...")
            clinical_texts = [str(text) if text is not None else "" for text in df['clinical_text'].tolist()]
            
            if config['model_type'] == 'standard':
                clinical_embeddings = self.generate_standard_embeddings(
                    config['model_name'], clinical_texts, 
                    config['max_length'], config['batch_size']
                )
            elif config['model_type'] == 'llama':
                clinical_embeddings = self.generate_llama_embeddings(
                    config['model_name'], clinical_texts,
                    config['max_length'], config['batch_size']
                )
            elif config['model_type'] == 'medgemma':
                clinical_embeddings = self.generate_medgemma_embeddings(
                    config['model_name'], clinical_texts
                )
            
            self.save_embeddings(
                clinical_embeddings, 
                df['case_submitter_id'].tolist(),
                df['project_id'].tolist(),
                model_key, 'clinical'
            )
        else:
            print(f"Clinical embeddings for {model_key} already exist, skipping...")
        
        # Process pathology texts
        if not self.check_existing_embeddings(model_key, 'pathology'):
            print(f"\nGenerating pathology embeddings for {model_key}...")
            pathology_texts = [str(text) if text is not None else "" for text in df['pathology_text'].tolist()]
            
            if config['model_type'] == 'standard':
                pathology_embeddings = self.generate_standard_embeddings(
                    config['model_name'], pathology_texts,
                    config['max_length'], config['batch_size']
                )
            elif config['model_type'] == 'llama':
                pathology_embeddings = self.generate_llama_embeddings(
                    config['model_name'], pathology_texts,
                    config['max_length'], config['batch_size']
                )
            elif config['model_type'] == 'medgemma':
                pathology_embeddings = self.generate_medgemma_embeddings(
                    config['model_name'], pathology_texts
                )
            
            self.save_embeddings(
                pathology_embeddings,
                df['case_submitter_id'].tolist(),
                df['project_id'].tolist(),
                model_key, 'pathology'
            )
        else:
            print(f"Pathology embeddings for {model_key} already exist, skipping...")
    
    def save_patient_data(self, df):
        """Save patient data CSV if it doesn't exist"""
        patient_data_file = os.path.join(self.output_dir, "patient_data.csv")
        if not os.path.exists(patient_data_file):
            # Print available columns for debugging
            print("Available columns:", df.columns.tolist())
            
            # Map column names from the TCGASurvivalProcessor output
            patient_df = df[['case_submitter_id', 'project_id', 'survival_time_days', 'survival_status']].copy()
            patient_df = patient_df.rename(columns={
                'case_submitter_id': 'patient_id',
                'survival_time_days': 'overall_survival_time',
                'survival_status': 'overall_survival_event'
            })
            patient_df.to_csv(patient_data_file, index=False)
            print(f"Saved patient data to {patient_data_file}")
    
    def generate_all_embeddings(self):
        """Generate embeddings for all models"""
        # Load data
        df = self.load_data()
        
        # Save patient data
        self.save_patient_data(df)
        
        # Generate embeddings for each model
        for model_key in self.models_config.keys():
            print(f"\n{'='*60}")
            print(f"Processing {model_key.upper()} model")
            print(f"{'='*60}")
            
            try:
                self.generate_embeddings_for_model(model_key, df)
            except Exception as e:
                print(f"Error generating embeddings for {model_key}: {e}")
                continue
        
        print("\n✅ All embeddings generation complete!")
        print(f"Output directory: {self.output_dir}")
        
        # List generated files
        print("\nGenerated files:")
        for file in sorted(os.listdir(self.output_dir)):
            if file.endswith('.pkl') or file.endswith('.csv'):
                file_path = os.path.join(self.output_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  - {file} ({file_size:.2f} MB)")


def main():
    """Main function to generate all embeddings"""
    generator = UnifiedEmbeddingsGenerator()
    generator.generate_all_embeddings()


if __name__ == "__main__":
    main()