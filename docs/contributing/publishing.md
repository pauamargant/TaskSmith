# Publishing the documentation

The TaskSmith documentation is a standard **MkDocs Material** site (`mkdocs.yml`, `docs/`).

## Current status

- **Local / CI:** `mkdocs build --strict` validates that the site builds.
- **Public hosting:** not wired by default. Choose one when the content baseline is stable.

## Options (deferred gate)

| Option | When it fits |
| --- | --- |
| **Read the Docs** | Automated builds from `main`, versioned docs, simple MkDocs support. Add `.readthedocs.yaml` pointing at this repo and doc dependencies (`uv` or `pip` install of `docs` group). |
| **GitHub Pages** | Keep docs in-repo; add a workflow that runs `mkdocs build` and publishes `site/` to `gh-pages` or Pages artifacts. |

## Local commands

```bash
uv sync --group docs
uv run mkdocs serve
uv run mkdocs build --strict
```

Record the chosen platform and any secrets (RTD webhook, Pages branch) in your team runbook when you enable it.
