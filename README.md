# File Retrieval Experiment

This project runs an experiment that compares different file retrieval methods (Sequential, Direct/Hash, Binary, Cached) and generates performance metrics and visualizations.

## What this does
- Runs timed trials across configurable dataset sizes.
- Measures average/variance/min/max/95th-percentile latency per method.
- Exports results to `performance_data.csv` and `results.json`.
- Generates a set of visualization PNGs (executive summary and detailed charts).

## Generated files
- `performance_data.csv` — raw results CSV
- `results.json` — full results in JSON
- `00_executive_summary.png` — single-page executive summary
- `01_performance_comparison.png` … `09_advanced_metrics.png` — detailed visualizations
- `animation_frame_*.png` — optional animation frames when generated

## Dependencies
Recommended (install via pip):
- Python 3.8+ (higher is fine)
- `matplotlib`
- `numpy`
- `seaborn` (optional; the code will fall back to matplotlib style if seaborn is not installed)

A minimal `requirements.txt` is included.

## Install (PowerShell)
```powershell
python -m pip install -r "c:\Users\angel\Downloads\fileretrievalexp\requirements.txt"
```

> Note: `seaborn` is optional. If it isn't installed the visualization module will still run using matplotlib fallbacks.

## Running the experiment (PowerShell)
Run the main script and follow the interactive menu:

```powershell
python "c:\Users\angel\Downloads\fileretrievalexp\file_retrieval_experiment.py"
```

The script will prompt you to choose a test configuration. After the run it will export the CSV/JSON and save PNG visualizations in the current directory.

## Changes made
- `visualization.py`: made the `seaborn` import optional and added a guarded call to `sns.set_style()` so the program won't crash if `seaborn` is not installed. This keeps visualizations working with matplotlib alone.
- `requirements.txt`: added recommended packages for convenience.

## Troubleshooting
- If you see a `ModuleNotFoundError` for `matplotlib` or `numpy`, install them using pip as shown above.
- For the best visuals, install `seaborn`.

## Next steps (optional)
- Add a CLI flag to disable seaborn styling explicitly.
- Add unit tests that mock `results` and verify the visualization functions run without error.

If you want, I can add one of these next steps now. 
