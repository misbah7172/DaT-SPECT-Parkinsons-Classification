# DaT Parkinson's Challenge — Code Submission Format

Source: https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/

**This is a code execution challenge!** Rather than submitting your predicted labels, you'll package everything needed to perform inference and submit a ZIP archive for containerized execution. The runtime repository (https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime) contains the complete specification for the runtime and gives you tools for local testing.

**All submissions for inference must run on Python 3.12. No other languages or versions of Python are supported.** If you want to learn more about how our code execution competitions work, check out the blog post: https://drivendata.co/blog/code-execution-competitions/

The general process for making a submission is:

* Create a properly formatted `submission.zip` containing your model and code for performing inference
* Debug and verify your submission works correctly by:

  * Testing your submission locally using Docker
  * Submitting as a smoke test to test against a small dataset, from the submissions page
* Submit as a normal submission to predict on the full test set, also using the submissions page

Results of your submissions are tracked on the submissions page. You can track the execution status of your submissions here, view logs (note: log output is limited to 300 lines, with 300 characters per line), and track the score of your predictions. Errors that happen during scoring will also appear here.

## What to submit

Your submission will be a ZIP archive (e.g., `submission.zip`). The root level of the `submission.zip` file **must** contain a `main.py` Python script which performs inference in the competition execution environment and writes your predictions to the required output file.

Example of what a submission might look like. The only file that must exist in `submission.zip` is `main.py`. Note that you should make sure `main.py` is in the root layer of the archive — zipping up a folder containing `main.py` often causes it to be nested inside the folder instead.

```
submission.zip
├── main.py                 # Required: entrypoint script (executed by the evaluator)
├── my\_module/
│   ├── preprocessing.py
│   ├── model.py
│   └── ...
└── my\_model\_weights.bin
```

Runtime repository: https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime
Example submission: https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime/tree/main/examples/minimal

## What happens during execution

During code execution, your submission will be unzipped and run in our cloud compute cluster. The container will run your `main.py` script. The script will have access to the following directory structure:

```
/code\_execution/
├── data/
│   ├── niftis/
│   │   └── ...                  # NIfTI imaging files for the test set
│   └── submission\_format.csv
└── src/
│   └── ...your files are unzipped into here...
└── submission.csv - you should write your predictions here
```

The working directory will be `/code\_execution/`. Your code will be unzipped into `src/` and the test set DaT scans will be in the `data/niftis/` subfolder. The `data/` directory is read-only.

## Output format

Your code must write a `submission.csv` file to the root folder containing one prediction per examination. The format must match the format of `submission\_format.csv` in the runtime exactly, with only two columns:

* `uid` (str) — unique identifier for each DaT scan examination
* `is\_pathologic` (float) — your predicted probability that the examination is abnormal, as a value between `0.0` and `1.0`

Predictions are scored with log loss (see the Problem Description page), so submit calibrated probabilities rather than hard `0`/`1` labels.

Example predictions:

|uid|is\_pathologic|
|-|-|
|xaji0y6d|0.04|
|pbhsahxt|0.12|
|hv3a3zmf|0.87|
|8mdd4v30|0.55|
|t9nt3w5u|0.98|

Your `submission.csv` file would look like:

```
uid,is\_pathologic
xaji0y6d,0.04
pbhsahxt,0.12
hv3a3zmf,0.87
8mdd4v30,0.55
t9nt3w5u,0.98
```

The submission should be written to `submission.csv` in the root folder of the working directory. Use the `submission\_format.csv` in the execution environment as a template, as illustrated in the runtime repo examples: https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime/tree/main/examples/minimal

## Testing your submission

The machines for executing your inference code are a shared resource across competitors, so please be conscientious in your use of them. Before you make a full submission, you should:

1. Test locally
2. Test in the limited smoke test environment
3. Once your submission runs successfully both locally and in the smoke test environment, make a full competition submission

### Testing your submission locally

**You should first and foremost test your submission locally using Docker.** This is a great way to work out any bugs and ensure that your model performs inference successfully. See the runtime repository's README for further instructions: https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime

### Smoke tests

For additional debugging, we provide a "smoke test" environment that replicates the test inference runtime but runs only on a small set of examinations. In the smoke test environment, the test data structure is the same (`niftis/` and `submission\_format.csv`), but the available data represents a small sample drawn from the training set rather than the full test set. Smoke tests are not considered for prize evaluation and are intended to let you test your code for correctness and speed.

## Submission checklist

* Submission includes `main.py` in the root directory of the ZIP archive. There can be additional Python modules if needed — see the "What to submit" section above.
* Submission contains any model weights that need to be loaded. **There will be no network access inside the runtime environment.**
* Submission does not print or log any information about the test dataset. Doing so may be grounds for disqualification based on the requirement that test samples are processed independently (see Rules — Use of test data).
* The `main.py` script loads the data for inference from the `data/` subdirectory of the working directory. This directory is read-only.
* The `main.py` script writes predictions to `submission.csv` in the root folder. The format of this file must match `submission\_format.csv` exactly.

Please be aware that the machines for executing your code are a shared resource across competitors, so please be conscientious in your use of them. Thoroughly test your submissions locally, add progress information to your logs, and cancel jobs that will fail because you expect them to run into the time limit.

## Runtime environment and constraints

Your code will be executed within a container whose image is defined in the runtime repository (https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime). The limits are as follows:

* Your submission must be able to run using Python 3.12 and the Python dependencies defined in the runtime repository. Dependencies are specified in the `pyproject.toml` and `uv.lock` files.
* The submission must complete execution in 3 hours or less. Smoke test submissions must complete in 6 minutes or less.
* The container will have access to the following hardware:

  * A single NVIDIA A100 GPU with 80 GiB of GPU VRAM
  * 24 vCPUs
* The container has access to 24 vCPUs powered by an AMD EPYC™ 7V13 chip and 220GB RAM.
* The container has 1 NVIDIA A100 GPU with 80GB VRAM. All of your code should run within the GPU environments in the container, even if actual computation happens on the CPU.
* The container will **not** have network access. All necessary files (code and model assets) must be included in your submission.
* The container will **not** have root access to the filesystem.
* Logging display is limited to 300 lines and 300 characters in each line.

## Requesting additional package installations to runtime

Since the container will not have network access, all packages must be pre-installed. The runtime is built using uv (https://docs.astral.sh/uv/) and PyTorch (https://pytorch.org/) backed by CUDA 12.9. Package additions will be considered as long as they do not conflict and can build successfully. Python packages should be available via PyPI. To request an additional package be added to the runtime environment, follow the instructions in the runtime repository.

