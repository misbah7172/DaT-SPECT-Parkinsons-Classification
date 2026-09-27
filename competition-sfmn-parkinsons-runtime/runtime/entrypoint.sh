#!/bin/bash

set -euxo pipefail

log() {
    set +x
    local level="$1"; shift
    printf '%s | %-5s | %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S.%3N')" \
        "$level" \
        "$*"
    set -x
}

main () {
    expected_filename=main.py

    cd /code_execution

    submission_files=$(zip -sf ./submission/submission.zip)
    if ! grep -q ${expected_filename}<<<$submission_files; then
        log ERROR "Submission zip archive must include $expected_filename"
    return 1
    fi

    log INFO "Unpacking submission"
    unzip ./submission/submission.zip -d ./

    ls -alh

    if [ -d "data" ]; then
        log INFO "Data directory contents:"
        ls -alh data/
    else
        log WARN "No data directory found at ./data"
    fi

    if [ -n "${IS_SMOKE_TEST:-}" ]; then
        log INFO "Smoke test mode enabled (IS_SMOKE_TEST=${IS_SMOKE_TEST})"
    fi

    echo "Running submission..."
    python main.py

    echo "Exporting submission.csv result..."

    # Valid scripts must create a "submission.csv" file within the same directory as main
    if [ -f "submission.csv" ]; then
        log INFO "Script completed its run."
        cp submission.csv ./submission/submission.csv
    else
        log ERROR "Script did not produce a submission.csv file in the main directory."
        return 1
    fi
}

main |& tee "/code_execution/submission/log.txt"
exit_code=${PIPESTATUS[0]}

# Copy for terminationMessagePath
cp /code_execution/submission/log.txt /tmp/log

log INFO "Submission run completed with exit code: $exit_code" | tee -a "/code_execution/submission/log.txt" "/tmp/log"

exit $exit_code
