import torch
import joblib

def save_artifacts(model, meta, model_path="T20-WC_model.pth", prep_path="T20-WC_model_preprocess.joblib"):
    """
    model: your trained embedding model (Win_Pred_Model )
    meta: dict returned by build_inputs() containing team_le, city_le, scaler, n_teams, n_cities, device
    """

    # 1) Save model weights
    torch.save(model.state_dict(), model_path)

    # 2) Save preprocessing objects + config needed to rebuild model
    pack = {
        "team_le": meta["team_le"],
        "city_le": meta["city_le"],
        "scaler": meta["scaler"],
        "n_teams": meta["n_teams"],
        "n_cities": meta["n_cities"],

        # also save model hyperparams YOU used to create the model
        # (edit these to match your Win_Pred_Model __init__)
        "team_emb_dim": getattr(model, "team_emb", None).embedding_dim if hasattr(model, "team_emb") else None,
        "city_emb_dim": getattr(model, "city_emb", None).embedding_dim if hasattr(model, "city_emb") else None,
    }
    joblib.dump(pack, prep_path)

    print(f"Saved model -> {model_path}")
    print(f"Saved preprocess -> {prep_path}")

