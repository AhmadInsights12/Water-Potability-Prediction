# Water potability prediction

A Streamlit application for exploring the project's water-quality dataset and
predicting potability with the trained Random Forest model in
`water_model.pkl`.

## Run in Visual Studio Code

1. Open this folder in Visual Studio Code.
2. Select the project's `.venv` Python interpreter.
3. In the integrated PowerShell terminal, run:

   ```powershell
   uv sync
   uv run streamlit run main.py
   ```

4. Open the local URL printed by Streamlit in your browser.

If `uv` is not installed, install the dependencies from `pyproject.toml` into
the selected virtual environment, then run `streamlit run main.py`.

## App features

- **Overview:** dataset counts, class balance, expected measurements, and the
  model defaults used to fill missing values.
- **Single sample:** enter nine water measurements and view the model's
  predicted class and estimated potable probability.
- **Batch predictions:** upload a CSV containing the model measurement
  columns, review predictions, and download the results.

The reusable prediction and model-validation logic is in `water_backend.py`.
The Streamlit interface is in `main.py`.

## Run the backend tests

```powershell
uv run python -m unittest discover -s tests -v
```

## Important limitation

This is an educational machine-learning project. A model prediction does not
certify that water is safe to drink and must not replace laboratory testing or
guidance from a qualified water-quality professional.
