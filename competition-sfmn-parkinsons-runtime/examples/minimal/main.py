# This is a minimal example of how to load the data and generate predictions for the test set.
# This example simply saves the submission format but demonstrates how to load in model assets and source code included in a submission
# To test this submission, run:
# $ `just pack-example minimal`
# $ `just test-submission`
from pathlib import Path

import nibabel
from loguru import logger
import pandas as pd

from src.model import DATScanModel


DATA_DIR = Path("/code_execution/data")
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")
SRC_ROOT = Path(__file__).parent.resolve()


def main() -> None:
    # Load data
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    logger.info(f"Loaded submission_format.csv.")

    # Check that one image file exists
    example_img_id = submission_format.loc[0, "uid"]
    example_img_path = NIFTI_DIR / f"{example_img_id}.nii.gz"
    example_img = nibabel.load(example_img_path)
    logger.info(f"Loaded example image from {example_img_path}.")

    # Demonstrate loading in a model
    model_path = SRC_ROOT / "model" / "model.txt"
    logger.info(f"Loading model from: {model_path}")
    model = DATScanModel.load(model_path)

    # Demonstrate filling submission_format with predictions
    logger.info("Generating predictions for nifti images.")
    for uid in submission_format["uid"]:
        img_path = NIFTI_DIR / f"{uid}.nii.gz"
        img = nibabel.load(img_path)
        # Predict with your model. Redacted from example submission.
        # submission_format.loc[submission_format["uid"] == uid, "is_pathologic"] = (
        #    model.predict(img)
        # )

    # save as "submission.csv" in the root folder, where it is expected
    logger.info("Writing out submission.csv")
    submission_format.to_csv(WRITE_SUBMISSION_PATH, index=False)
    logger.success(
        f"Predictions for {len(submission_format):,} responses were written to {WRITE_SUBMISSION_PATH}"
    )


if __name__ == "__main__":
    main()
