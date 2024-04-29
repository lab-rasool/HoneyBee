import os

import minds
from dotenv import load_dotenv

load_dotenv()


def download(query=None, project_folder=None, include=None, exclude=None):
    manifest_file = project_folder + "\\manifest.json"
    query_cohort = minds.build_cohort(
        query=query,
        output_dir=project_folder,
        manifest=manifest_file if os.path.exists(manifest_file) else None,
    )
    query_cohort.download(threads=8, include=include, exclude=exclude)


def main():
    query = """
    SELECT * FROM minds.clinical WHERE project_id IN ('TCGA-LUAD', 'TCGA-LUSC')
    """
    if not os.path.exists("D:\\lung"):
        os.makedirs("D:\\lung")
    download(
        query=query,
        project_folder="D:\\lung",
        include=["Slide Image", "Pathology Report"],
    )


if __name__ == "__main__":
    main()
