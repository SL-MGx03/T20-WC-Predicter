import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from dataframe import df
from Win_Pred_Model import Win_Pred_Model
from features import accuracy_fn
from save_model import save_artifacts

def build_inputs(df, train_size=0.8, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_local = df.copy()
    df_local.columns = df_local.columns.astype(str).str.strip()
    df_clean = df_local[df_local["Winner"] != "tie"].copy()

    # teams
    for col in ["Team A", "Team B"]:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # city cleanup
    df_clean["City"] = df_clean["City"].fillna("Unknown")
    df_clean["City"] = (
        df_clean["City"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    )
    city_alias = {"bengaluru": "bangalore", "benagluru": "bangalore", "mumbai (wankhede)": "mumbai"}
    df_clean["City"] = df_clean["City"].replace(city_alias)

    # encoders
    all_team_names = pd.concat([df_clean["Team A"], df_clean["Team B"]]).unique()
    team_le = LabelEncoder().fit(all_team_names)
    city_le = LabelEncoder().fit(df_clean["City"].unique())

    team_a = team_le.transform(df_clean["Team A"]).astype(np.int64)
    team_b = team_le.transform(df_clean["Team B"]).astype(np.int64)
    city = city_le.transform(df_clean["City"]).astype(np.int64)

    num = df_clean[["First Inning", "First Inn Wickets"]].astype("float32").to_numpy()
    y = (df_clean["Winner"] == df_clean["Team A"]).astype("float32").to_numpy()

    # split indices (stratify keeps win/loss ratio similar)
    idx = np.arange(len(df_clean))
    idx_train, idx_test = train_test_split(idx, train_size=train_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    num_train = scaler.fit_transform(num[idx_train]).astype("float32")
    num_test = scaler.transform(num[idx_test]).astype("float32")

    # torch
    team_a_train = torch.tensor(team_a[idx_train], dtype=torch.long, device=device)
    team_b_train = torch.tensor(team_b[idx_train], dtype=torch.long, device=device)
    city_train = torch.tensor(city[idx_train], dtype=torch.long, device=device)
    num_train_t = torch.tensor(num_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y[idx_train], dtype=torch.float32, device=device)

    team_a_test = torch.tensor(team_a[idx_test], dtype=torch.long, device=device)
    team_b_test = torch.tensor(team_b[idx_test], dtype=torch.long, device=device)
    city_test = torch.tensor(city[idx_test], dtype=torch.long, device=device)
    num_test_t = torch.tensor(num_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y[idx_test], dtype=torch.float32, device=device)

    meta = {
        "team_le": team_le,
        "city_le": city_le,
        "scaler": scaler,
        "n_teams": len(team_le.classes_),
        "n_cities": len(city_le.classes_),
        "device": device,
    }

    train_tensors = (team_a_train, team_b_train, city_train, num_train_t, y_train_t)
    test_tensors = (team_a_test, team_b_test, city_test, num_test_t, y_test_t)

    return train_tensors, test_tensors, meta



def train_model(model, train_tensors, test_tensors, epochs=200, lr=1e-3):
    team_a_tr, team_b_tr, city_tr, num_tr, y_tr = train_tensors
    team_a_te, team_b_te, city_te, num_te, y_te = test_tensors

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    for epoch in range(epochs):
        model.train()
        logits = model(team_a_tr, team_b_tr, city_tr, num_tr)
        preds = torch.round(torch.sigmoid(logits))
        loss = loss_fn(logits, y_tr)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.inference_mode():
            t_logits = model(team_a_te, team_b_te, city_te, num_te)
            t_preds = torch.round(torch.sigmoid(t_logits))
            t_loss = loss_fn(t_logits, y_te)

            # your accuracy_fn should work if it expects tensors
            tr_acc = accuracy_fn(y_true=y_tr, y_pred=preds)
            te_acc = accuracy_fn(y_true=y_te, y_pred=t_preds)

        if epoch % 20 == 0:
            print(f"Epoch: {epoch} | Train loss: {loss:.4f} | Test loss: {t_loss:.4f}")
            print(f"Train Accuracy: {tr_acc:.2f} | Test Accuracy: {te_acc:.2f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train_tensors, test_tensors, meta = build_inputs(df)
    model_0 = Win_Pred_Model(n_teams=meta["n_teams"], n_cities= meta["n_cities"]).to(device)
    train_model(model_0, train_tensors, test_tensors)

   
    """ Use this command one time to save the model to local"""
    #save_artifacts(model_0,meta) 

