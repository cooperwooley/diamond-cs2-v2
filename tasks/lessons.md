# Lessons

## Lessons
- **No AI attribution in this repo.** Commits, PRs, issues, and files carry no co-author trailers, session
  links, or tool/assistant mentions. Everything is authored by the repo owner. Check commit messages and
  PR bodies before pushing.
- **Keep the tree clean at the end of every working session.** A local auto-save hook commits and pushes dirty
  trees with its own message; commit deliberately first.
- **ruff skips its EXE rules under WSL**, so local ruff can pass while CI fails. Executable scripts need a shebang.
- **Float ties in binning:** never let float rounding decide a tie. Quantise values and edges before
  `searchsorted`, and test with an exact (`Fraction`) reference over integer inputs.
- **Casting float to uint8 in torch wraps, it doesn't clamp** (257.5 -> 1). Always `.clamp(0, 255)` before `.byte()`.
