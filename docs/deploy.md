# Deploying the web app, and recording the demo GIF

## Streamlit Community Cloud

The repository is already laid out for it: `app/streamlit_app.py` (entry point),
`requirements.txt` (installs `.[app]`), `packages.txt` (`libgomp1`, the OpenMP runtime
LightGBM's Linux wheel needs) and `.streamlit/config.toml` (50 MB uploads, no usage
statistics).

1. Push `main` to GitHub (the app is deployed from a branch; the tag `v1.1.0` marks the
   release).
2. Sign in at https://share.streamlit.io with the GitHub account that owns the repository
   and allow it to read the repository.
3. **Create app** -> "Deploy a public app from GitHub":
   - Repository: `iamrohitkumaryadav9/holdout`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - App URL: choose one (for example `holdout-agent`).
4. **Advanced settings** -> Python version **3.12** (the package requires `>=3.12`).
   Secrets: none are needed. The app never uses a server-side Gemini key: visitors who
   choose Gemini paste their own key, which stays in their session. Optionally set
   `HOLDOUT_PROJECT_CAP_INR = "100"` to lower the server-wide spend cap that the ledger
   enforces across all visitors' Gemini runs (default Rs 750).
5. **Deploy.** The first build installs the pinned stack (a few minutes). Check the build
   log for `Successfully installed ... holdout-1.1.0 ... streamlit-1.67.0`.
6. Smoke check in the browser:
   - The sidebar shows **Holdout 1.1.0**, and the link opens the `v1.1.0` tag.
   - Run page: demo "Leaky column (direct_leak)", Offline -> **Run**. You should see a red
     **Critical** block (the leak, resolved), an orange **Warnings** block with the
     "need your judgement" note, and three metrics.
   - Benchmark page: "Pre-registered results (v1.0.0)" comes first, then "Post-hoc
     exploratory ...", with the trade-offs.
7. If the build fails on `pip install -r requirements.txt`, check that the log runs it
   from the repository root (`.[app]` is relative to it). Otherwise replace the line with
   `holdout[app] @ git+https://github.com/iamrohitkumaryadav9/holdout@v1.1.0`.

Limits to know: the free tier has about 1 GB of memory and sleeps when idle. Large
uploads or the `group_leakage` demo (a follow-up run) take a minute or two. Every run
writes to a temporary directory that is gone on restart; downloads (report, manifest,
`pipeline.py`) are the way to keep a run.

## Demo GIF checklist

Target: 20-30 s, 1280x720 (or 1200 wide), under 8 MB, no audio. Record with ScreenToGif
(Windows) or Kap (macOS), at 12-15 fps.

Before recording:

- [ ] Use the deployed app (or `streamlit run app/streamlit_app.py`) in a clean browser
      window at 100% zoom, light theme, bookmarks bar hidden.
- [ ] **Offline** selected; no API key anywhere on screen (not even masked).
- [ ] Run each demo once beforehand, so imports and the cache are warm.
- [ ] Close notifications and other tabs; nothing personal in the window title or URL bar.

Shot list:

1. [ ] (3 s) Run page: the one-line description; select demo **Leaky column
       (direct_leak)**; the preview table shows `refund_issued`.
2. [ ] (4 s) Target `converted`, Planner **Offline** -> click **Run**; the status box shows
       progress lines.
3. [ ] (6 s) Scroll to **Problems found before planning**: the red **Critical** block
       (`refund_issued`, status *resolved*) above the orange **Warnings** block with its
       "warnings need your judgement" note.
4. [ ] (4 s) The three metrics (validation, dummy, sealed holdout) and the green "holdout
       was opened once" line.
5. [ ] (5 s) Switch demo to **Repeated customers (group_leakage)** -> Run: the red **Holdout
       invalid** banner, then the follow-up run with the corrected split.
6. [ ] (5 s) Benchmark page: the "Pre-registered results (v1.0.0)" header with
       51.8% [33.9%, 69.6%], then scroll to the post-hoc header and the trade-offs.

After recording:

- [ ] Trim dead frames and reduce colours (ScreenToGif: "Save as" GIF, quantizer
      "Octree", 128-256 colours); check the size is under 8 MB.
- [ ] Watch it once: every number on screen must match docs/results.md or the run shown.
- [ ] Save as `docs/img/demo.gif` and add `![Holdout demo](docs/img/demo.gif)` under the
      first paragraph of the README.
