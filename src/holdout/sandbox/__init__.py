"""EDA sandbox: run short, read-only analysis snippets on a copy of development rows.

This is a damage limiter, not a security boundary. It stops accidents and casual misuse by
generated code (reading other files, writing outside its directory, network calls, runaway
time or memory) with layered, best-effort controls: a static AST allowlist, a restricted
runtime (pruned builtins and an import guard), a fresh isolated interpreter in an empty
temporary directory with a minimal environment (no API key), a timeout, a memory limit and
an output cap. A determined attacker with arbitrary Python can defeat in-process
restrictions; do not run untrusted code here that you would not run on your own machine.
"""
