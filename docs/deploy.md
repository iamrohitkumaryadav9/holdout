# Deploying the web app

The app is live at **https://holdout.streamlit.app**, deployed on Streamlit Community Cloud
from `main` (it redeploys on every push). The steps below set up the same deployment for
a fork or a fresh account.

## Streamlit Community Cloud

The repository is already laid out for it: `app/streamlit_app.py` (entry point),
`requirements.txt` (installs `.[app]`), `packages.txt` (`libgomp1`, the OpenMP runtime
LightGBM's Linux wheel needs) and `.streamlit/config.toml` (50 MB uploads, no usage
statistics).

1. Push `main` to GitHub (the app is deployed from a branch; the tag `v1.1.4` marks the
   release).
2. Sign in at https://share.streamlit.io with the GitHub account that owns the repository
   and allow it to read the repository.
3. **Create app** -> "Deploy a public app from GitHub":
   - Repository: `iamrohitkumaryadav9/holdout`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - App URL: `holdout` (the live app is https://holdout.streamlit.app; a fork needs its
     own name).
4. **Advanced settings** -> Python version **3.12** (the package requires `>=3.12`).
   Secrets: none are needed. The app never uses a server-side Gemini key: visitors who
   choose Gemini paste their own key, which stays in their session. Optionally set
   `HOLDOUT_PROJECT_CAP_INR = "100"` to lower the server-wide spend cap that the ledger
   enforces across all visitors' Gemini runs (default Rs 750).
5. **Deploy.** The first build installs the pinned stack (a few minutes). Check the build
   log for `Successfully installed ... holdout-1.1.4 ... streamlit-1.67.0`.
6. Smoke check in the browser:
   - The sidebar shows **Holdout 1.1.4**, and the link opens the `v1.1.4` tag.
   - Run page: demo "Leaky column (direct_leak)", Offline -> **Run**. You should see a red
     **Critical** block (the leak, resolved), an orange **Warnings** block with the
     "need your judgement" note, and three metrics.
   - Benchmark page: "Pre-registered results (v1.0.0)" comes first, then "Post-hoc
     exploratory ...", with the trade-offs.
7. If the build fails on `pip install -r requirements.txt`, check that the log runs it
   from the repository root (`.[app]` is relative to it). Otherwise replace the line with
   `holdout[app] @ git+https://github.com/iamrohitkumaryadav9/holdout@v1.1.4`.

Limits to know: the free tier has about 1 GB of memory and sleeps when idle. Large
uploads or the `group_leakage` demo (a follow-up run) take a minute or two. Every run
writes to a temporary directory that is gone on restart; downloads (report, manifest,
`pipeline.py`) are the way to keep a run.
