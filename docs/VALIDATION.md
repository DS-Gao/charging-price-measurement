# Release validation

Checked locally on 18 October 2026 with Python 3.12.14 on Windows and the versions in `requirements-lock.txt`.

| Check | Outcome and scope |
|---|---|
| Numerical contracts | Four unit tests passed: weighted ties and scaling, zero-weight filtering, missing-versus-zero comparisons, and known-truth reconstruction invariants. |
| Core estimates | Derived-input workflow passed its reference comparisons, including exact membership/count checks and numerical tolerances specified in the reports. |
| New resampling | All 2,000 shared-station bootstrap repetitions regenerated; 390,000 stored result rows matched the reference draws. |
| Distribution analyses | All four workflow stages completed, including component/weight comparisons, signed and threshold analyses, and amount/screening summaries. |
| Raw reconstruction | Both public transaction files were read locally and the endpoint reconstruction passed its packaged reference comparisons. Raw files are downloaded separately. |
| Synthetic example | All 120 scenario-repetition runs completed (15 scenarios × 8 repetitions). This is an execution check, not a new coverage study. |
| Relocation | Core and distribution workflows passed from a temporary directory while a Python file-access audit hook denied access to the original project except its dependency environment. The guard was tested explicitly; no unexpected access attempts occurred. This is not an operating-system sandbox. |

Machine-readable checks are in [checks/](checks/). Runtime paths have been omitted from public records. Data/reference files retain their numerical content; organization and workflow paths changed.

The automated workflow in `.github/workflows/verify.yml` repeats file verification, numerical tests, core/distribution runs and the small simulation on Windows and Linux. Consult the repository's Actions page for the actual status of each run; adding the workflow does not itself establish a pass.

These checks do not rerun every historical analysis or establish real latent-price accuracy, full-process interval coverage, or coverage of the original variance-share model. See [REPRODUCTION.md](REPRODUCTION.md).
