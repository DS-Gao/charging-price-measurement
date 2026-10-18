# Maintaining the release

Keep statistical changes separate from packaging changes. For a change to an input, document its scientific reason, recompute affected outputs and review the numerical comparisons before updating any checksum. Do not update hashes merely to silence a failing comparison.

Report problems through the repository issue tracker with the command, Python/package versions, error message and relevant result filenames. Do not attach private transaction records or credentials.

Repository preparation follows common scientific-code release practice: runnable entry points, dependency versions, test inputs, explicit parameters, provenance and reproducibility limits. See [Nature's code-publication guidance](https://www.nature.com/documents/GuidelinesCodePublication.pdf), [GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository), and [citation-file documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).
