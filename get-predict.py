# After running app.py and saving the model run this code seperately to get predictions

import torch
import joblib
import numpy as np

from Win_Pred_Model import Win_Pred_Model


def load_predictor(
    model_path: str = "T20-WC_model.pth",
    prep_path: str = "T20-WC_model_preprocess.joblib",
    device: str | None = None,
):
    """
    Loads preprocessing pack + rebuilds model architecture + loads weights.
    Returns (model, pack, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    pack = joblib.load(prep_path)

    model = Win_Pred_Model(
        n_teams=pack["n_teams"],
        n_cities=pack["n_cities"],
        team_emb_dim=pack.get("team_emb_dim") or 8,
        city_emb_dim=pack.get("city_emb_dim") or 6,
    ).to(device)

    # Load weights
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, pack, device


@torch.inference_mode()
def predict_win_probability(
    model,
    pack: dict,
    device,
    *,
    team_a: str,
    team_b: str,
    city: str,
    first_inning_runs: float,
    first_inning_wkts: float,
):
    """
    Returns:
      prob_team_a_wins (float in [0,1]),
      predicted_winner (str)
    """

    # --- cleanup  ---
    team_a = str(team_a).strip()
    team_b = str(team_b).strip()

    city = "Unknown" if city is None else str(city).strip().lower()
    city = " ".join(city.split())  # normalize whitespace
    city_alias = {"bengaluru": "bangalore", "benagluru": "bangalore", "mumbai (wankhede)": "mumbai"}
    city = city_alias.get(city, city)

    team_le = pack["team_le"]
    city_le = pack["city_le"]
    scaler = pack["scaler"]

    # --- encode categoricals ---
    try:
        team_a_id = team_le.transform([team_a])[0]
        team_b_id = team_le.transform([team_b])[0]
    except ValueError as e:
        raise ValueError(
            f"Unknown team name. Got team_a={team_a!r}, team_b={team_b!r}. "
            f"Known teams: {list(team_le.classes_)}"
        ) from e

    try:
        city_id = city_le.transform([city])[0]
    except ValueError as e:
        raise ValueError(
            f"Unknown city name. Got city={city!r}. "
            f"Known cities: {list(city_le.classes_)[:50]}{'...' if len(city_le.classes_) > 50 else ''}"
        ) from e

    # --- numeric features  ---
    num = np.array([[float(first_inning_runs), float(first_inning_wkts)]], dtype=np.float32)
    num_scaled = scaler.transform(num).astype(np.float32)

    # --- tensors ---
    t_team_a = torch.tensor([team_a_id], dtype=torch.long, device=device)
    t_team_b = torch.tensor([team_b_id], dtype=torch.long, device=device)
    t_city = torch.tensor([city_id], dtype=torch.long, device=device)
    t_num = torch.tensor(num_scaled, dtype=torch.float32, device=device)

    # --- forward ---
    logits = model(t_team_a, t_team_b, t_city, t_num)  # shape: (1,)
    prob = torch.sigmoid(logits).item()

    predicted = team_a if prob >= 0.5 else team_b
    return prob, predicted


if __name__ == "__main__":
    model, pack, device = load_predictor(
        model_path="T20-WC_model.pth",
        prep_path="T20-WC_model_preprocess.joblib",
    )

    #edit them for your need (team_a, team_b, city, first_inning_runs, first_inning_wkts)
    prob, winner = predict_win_probability(
        model,
        pack,
        device,
        team_a="Australia",
        team_b="Pakistan",
        city="Bangalore",
        first_inning_runs=140,
        first_inning_wkts=6,
    )

    print("Predicted winner:", winner)
    print(f"Prob(Team A wins): {round(prob*100, 2)} %")
    
