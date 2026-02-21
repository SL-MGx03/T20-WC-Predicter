# T20-WC-Predicter

A PyTorch-based cricket match winner prediction project.  
This repository builds a match-level dataset from JSON ball-by-ball files, trains a neural network with embeddings for categorical features (teams, city), and supports loading the saved model to run predictions. The project is structured so it can later be deployed as an API .

---

## Try it 

You can tryout our pre-trained model . 
- https://slmgx.live/cricket/t20

---

## Features

- **Dataset builder** from JSON match files (`data_set/*.json`) into a Pandas DataFrame.
- **Model training** using PyTorch:
  - Embeddings for **Team A**, **Team B**, and **City**
  - Numeric features: **First Inning runs** and **First Inning wickets**
- **Evaluation** with accuracy reporting during training.
- **Model persistence**:
  - Saves model weights (`.pth`)
  - Saves preprocessing artifacts (`LabelEncoder`, `StandardScaler`, etc.) via Joblib (`.joblib`)
- **Inference**:
  - Load saved artifacts and perform predictions on new inputs.

---

## Repository Structure

- `core.py` — Training pipeline (data cleaning, encoding, scaling, model training).
- `dataframe.py` — Builds the match-level DataFrame from JSON files.
- `json_return.py` — Collects JSON file paths and IDs from the dataset folder.
- `Win_Pred_Model.py` — PyTorch model definition (embeddings + MLP).
- `features.py` — Utility functions (accuracy function).
- `save_model.py` — Saves model weights and preprocessing artifacts.
- `graphs.py` — Generic Matplotlib plotting helper (optional utility).
- `get-predict.py` - Prediction using saved model
- `data_set/` — Folder containing match JSON files.

---

## Requirements

You will need Python 3.10+ (recommended) and the main dependencies:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `matplotlib` (optional, only if you use `graphs.py`)

Install dependencies (example):

```bash
pip install torch numpy pandas scikit-learn joblib matplotlib
```

---

## Dataset

Data Source: https://cricsheet.org/

Place match JSON files inside:

```text
data_set/
  match1.json
  match2.json
  ...
```

The dataset loader (`json_return.py`) reads all `*.json` files from `data_set/`.

---

## Training

Run the training script:

```bash
python core.py
```

Training does the following:

1. Loads the dataset DataFrame (`dataframe.py`)
2. Cleans values (teams, cities, handles missing city)
3. Encodes categorical features (teams, city)
4. Scales numeric features (`First Inning`, `First Inn Wickets`)
5. Trains `Win_Pred_Model`

---

## Saving the Model

In `core.py`, saving is currently commented:

```python
# save_artifacts(model_0, meta)
```

Uncomment it **after training** if you want to save the artifacts:

```python
save_artifacts(model_0, meta)
```

This produces:

- `T20-WC_model.pth`
- `T20-WC_model_preprocess.joblib`

---

## Inference (Load Model + Predict)

After saving, you can load the trained model and run predictions using an inference script (form `get-predict.py`).

A prediction typically requires:

- `team_a` (string)
- `team_b` (string)
- `city` (string)
- `first_inning_runs` (number)
- `first_inning_wkts` (number)

Output:
- Probability that **Team A wins**
- Predicted winner (Team A if probability ≥ 0.5 else Team B)

---

## Notes on Model Inputs

- The target label is:
  - `1` if `Winner == Team A`
  - `0` otherwise
- Rows where `Winner == "tie"` are removed during training.
- City names are normalized and a small alias map is applied (e.g. `bengaluru -> bangalore`).

---

## License

MIT License. See [`LICENSE`](LICENSE).
