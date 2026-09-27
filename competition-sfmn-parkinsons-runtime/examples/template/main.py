"""This is a template for the expected code submission format."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path("/code_execution/data")
NIFTI_DIR = DATA_ROOT / "niftis"
SUBMISSION_FORMAT_PATH = DATA_ROOT / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")


def predict(filename, rng):
    """Generate a random probability between 0 and 1."""
    return rng.random()


def main(seed=77):
    # load sumission format
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)

    # set starting random state for consistent runs
    rng = np.random.RandomState(seed)

    # iterate over all images in the metdata
    for uid in submission_format.index:
        # prepend data path to image name
        filepath = NIFTI_DIR / f"{uid}.nii.gz"

        # make sure the image exists
        assert filepath.exists()

        # generate a prediction for that file
        pred = predict(filepath, rng=rng)

        # assign to the right row in the submission format
        submission_format.loc[uid, "is_pathologic"] = pred

    # save as "submission.csv" in the root folder, where it is expected
    submission_format.to_csv(WRITE_SUBMISSION_PATH)


if __name__ == "__main__":
    main()
