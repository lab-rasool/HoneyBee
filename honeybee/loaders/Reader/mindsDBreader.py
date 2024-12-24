import json

import pandas as pd


def manifest_to_df(manifest_path, modality):
    """
    Converts a manifest JSON file into a pandas DataFrame for a specified modality.

    Args:
        manifest_path (str): The file path to the manifest JSON file.
        modality (str): The modality to extract from the manifest (e.g., 'CT', 'MRI').

    Returns:
        pd.DataFrame or None: A DataFrame containing the data for the specified modality,
                              with 'PatientID' and 'gdc_case_id' columns added. Returns
                              None if the modality is not found in any patient.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Initialize an empty DataFrame for the modality
    modality_df = pd.DataFrame()

    # Process each patient in the manifest
    for patient in manifest:
        patient_id = patient["PatientID"]
        gdc_case_id = patient["gdc_case_id"]

        # Check if the current patient has the requested modality
        if modality in patient:
            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(patient[modality])
            # Add 'PatientID' and 'gdc_case_id' columns
            df["PatientID"] = patient_id
            df["gdc_case_id"] = gdc_case_id

            # Append the new data to the existing DataFrame for this modality
            modality_df = pd.concat([modality_df, df], ignore_index=True)

    # Check if the modality DataFrame is not empty before returning
    if not modality_df.empty:
        return modality_df
    else:
        return None
