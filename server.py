"""
FastAPI ML Server

Provides authentication (signup/login/JWT), token accounting, model training,
prediction, and model management. Models are trained via facades from
`model_factory`, saved as joblib bundles under ./models, and metadata is stored
through the `dal` layer (SQLite).

Main endpoints:
- POST /signup, POST /login, POST /token
- GET  /health
- POST /train
- GET  /models
- POST /predict/classification
- POST /predict/regression
- POST /predict/by_id/{model_id}
- POST /predict/{model_name}     (use latest trained model of that name)
- GET  /tokens/{username}
- POST /add_tokens
- DELETE /remove_user
"""

from __future__ import annotations
import os, uuid, joblib
import dal
import jwt  # PyJWT
import pandas as pd
import tempfile, json
import sqlite3
import re
import math
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from typing import Any, Dict
from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException, Body, Depends, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel
from File_Converter_Factory import FileConverterFactory
from model_factory import ModelFactory
from errors import (
    AuthBadCredentials, AuthUserNotFound, AuthTokenExpired, AuthTokenInvalid,
    InputInvalidJSON, InputMissingColumns, InputModelParamsInvalid,
    TokensNotEnough, ModelNotFound, ModelKindMismatch, ModelTrainFailed,
    ModelPredictFailed, ModelBundleCorrupt, InternalError, log_and_raise
)
from enums import ModelName

# ---- Security / JWT ----
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

SECRET_KEY = "MY_SUPER_FIXED_SECRET_123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ---- App & DB bootstrap ----
dal.init_db()
app = FastAPI(title="ML Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---- Logging ----
logger = logging.getLogger("ml_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler("server.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
logger.info("Server starting up")

# ---- Schemas ----
class Signup(BaseModel):
    username: str
    password: str

class Login(BaseModel):
    username: str
    password: str

class RemoveUser(BaseModel):
    username: str
    password: str

class AddTokens(BaseModel):
    username: str
    credit_card: str
    amount: int

# ---- Helpers ----
def _valid_password(p: str) -> bool:
    return (
        isinstance(p, str) and
        len(p) >= 6 and
        re.search(r"[A-Z]", p) and
        re.search(r"[^A-Za-z0-9]", p)
    )

def create_access_token(data: dict, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=minutes)
    to_encode = data.copy()
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

from pathlib import Path
import tempfile

def _save_upload_to_temp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename).suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read()); tmp.flush()
    return Path(tmp.name)

def _load_df(upload: UploadFile) -> pd.DataFrame:
    p = _save_upload_to_temp(upload)
    converter = FileConverterFactory().get(p)
    csv_path = converter.convert_to_csv(p)
    return pd.read_csv(csv_path)

def _parse_json_field(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except Exception:
        log_and_raise(InputInvalidJSON(details={"value": s[:200]}))

def _clean_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            log_and_raise(AuthTokenInvalid())
    except jwt.ExpiredSignatureError:
        log_and_raise(AuthTokenExpired())
    except jwt.InvalidTokenError:
        log_and_raise(AuthTokenInvalid())
    user = dal.get_user(username)
    if not user:
        log_and_raise(AuthUserNotFound())
    return user

def _ensure_features(df: pd.DataFrame, features: list[str], label: str):
    missing = [c for c in features + [label] if c not in df.columns]
    if missing:
        log_and_raise(InputMissingColumns(details={"missing": missing}))

def _subset_df(df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    return df[features + [label]].copy()

# ---- Health ----
@app.get("/health")
def health():
    return {"ok": True}

# ---- Auth ----
@app.post("/signup")
def signup(payload: Signup):
    if not _valid_password(payload.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password too weak: min 6 chars, one uppercase, one special char."
        )
    try:
        dal.insert_user(payload.username, hash_password(payload.password), initial_tokens=15)
        dal.log_usage(payload.username, "signup", None, None, tokens_after_usage=15)
        logger.info(f"User '{payload.username}' registered")
        return {"message": "user created", "username": payload.username, "tokens": 15}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="username already exists")

@app.post("/login")
def login(payload: Login):
    user = dal.get_user(payload.username)
    if not user or not verify_password(payload.password, user["password_hash"]):
        log_and_raise(AuthBadCredentials())
    token = create_access_token({"sub": user["username"]})
    logger.info(f"User '{payload.username}' logged in")
    return {"access_token": token, "token_type": "bearer"}

@app.post("/token")
def issue_token(form: OAuth2PasswordRequestForm = Depends()):
    user = dal.get_user(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        log_and_raise(AuthBadCredentials())
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

# ---- Train ----
@app.post("/train")
def train_model(
    file: UploadFile = File(...),
    model_name: ModelName = Form(...),          # ← Enum => Dropdown
    features: str = Form(...),
    label: str = Form(...),
    model_params: str = Form("{}"),
    user: dict = Depends(get_current_user),
):
    """
    Train a model and persist its bundle to disk + metadata to DB.
    """
    try:
        # features: JSON array או קומות
        try:
            features_list = json.loads(features)
            if not isinstance(features_list, list):
                raise ValueError("features not a JSON array")
        except Exception:
            features_list = [c.strip() for c in re.split(r'[,\s]+', features) if c.strip()]
            if not features_list:
                raise HTTPException(
                    status_code=400,
                    detail='features must be JSON array (e.g. ["age","weight"]) or comma-separated (age,weight)'
                )

        try:
            params = json.loads(model_params) if model_params else {}
            if not isinstance(params, dict):
                raise ValueError("model_params not a JSON object")
        except Exception:
            raise HTTPException(status_code=400, detail="model_params must be a JSON object, e.g. {}")

        df = _load_df(file)
        _ensure_features(df, features_list, label)
        df = _subset_df(df, features_list, label)

        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=1,
                action="train",
                model_name=model_name.value,       # ← חשוב: .value
                file_name=file.filename,
            )
        except ValueError as e:
            logger.warning(f"User '{user['username']}' tried to train without enough tokens")
            raise HTTPException(status_code=402, detail=str(e))

        facade, kind = ModelFactory.create(model_name.value, params)  # ← חשוב: .value

        results = facade.train_and_evaluate(df, target_col=label)

        bundle = {
            "model_name": model_name.value,       # ← .value
            "kind": kind,
            "facade_class": facade.__class__.__name__,
            "model": getattr(facade, "model", None),
            "scaler": getattr(facade, "scaler", None),
            "poly": getattr(facade, "poly", None),
            "feature_cols": list(getattr(facade, "feature_cols", features_list)),
            "label": label,
        }

        model_id = str(uuid.uuid4())
        os.makedirs("models", exist_ok=True)
        pkl_path = os.path.join("models", f"{model_id}.pkl")
        joblib.dump(bundle, pkl_path)

        if kind == "classification":
            metrics = {"accuracy": _clean_float(results.get("accuracy"))}
        else:
            metrics = {
                "mse":  _clean_float(results.get("mse")),
                "rmse": _clean_float(results.get("rmse")),
                "r2":   _clean_float(results.get("r2")),
            }

        dal.insert_model(
            model_id=model_id,
            username=user["username"],
            model_name=model_name.value,         # ← .value
            kind=kind,
            path=pkl_path,
            features=features_list,
            label=label,
            metrics=metrics,
        )

        logger.info(
            f"User '{user['username']}' trained model='{model_name.value}' label='{label}' features={features_list}"
        )

        return {
            "status": "model trained",
            "model_id": model_id,
            "model_name": model_name.value,      # ← .value
            "kind": kind,
            "features": features_list,
            "label": label,
            "metrics": metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

# ---- List models ----
@app.get("/models")
def list_user_models(user: dict = Depends(get_current_user)):
    models = dal.list_models(user["username"])
    try:
        dal.update_tokens_and_log(
            username=user["username"],
            cost=1,
            action="list_models",
            model_name=None,
            file_name=None,
        )
        logger.info(f"User '{user['username']}' listed models")
    except ValueError:
        pass
    return {"models": models}

# ---- Predict – classification ----
@app.post("/predict/classification")
def predict_classification(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    model_name: ModelName = Form(ModelName.logreg),   # ← Enum
    data: str = Form("{}"),
    model_params: str = Form("{}"),
    user: dict = Depends(get_current_user),
):
    try:
        df = _load_df(file)
        new_row = _parse_json_field(data)
        params = _parse_json_field(model_params)

        facade, kind = ModelFactory.create(model_name.value, params)   # ← .value
        if kind != "classification":
            log_and_raise(ModelKindMismatch(details={"expected": "classification", "got": kind}))

        results = facade.train_and_evaluate(df, target_col=target_col)
        pred = facade.predict(pd.DataFrame([new_row]))

        if hasattr(pred, "iloc"):
            pred_value = pred.iloc[0]
        elif isinstance(pred, (list, tuple)):
            pred_value = pred[0]
        else:
            pred_value = pred

        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=5,
                action="predict/classification",
                model_name=model_name.value,      # ← .value
                file_name=file.filename,
            )
        except ValueError as e:
            if "not enough tokens" in str(e).lower():
                log_and_raise(TokensNotEnough())
            else:
                log_and_raise(InternalError(details={"cause": str(e)}))

        return {
            "model": model_name.value,           # ← .value
            "target_col": target_col,
            "prediction": str(pred_value),
            "metrics": {"accuracy": _clean_float(results.get("accuracy"))},
            "used_params": params,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

# ---- Predict – regression ----
@app.post("/predict/regression")
def predict_regression(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    model_name: ModelName = Form(ModelName.linear),   # ← Enum
    data: str = Form("{}"),
    model_params: str = Form("{}"),
    user: dict = Depends(get_current_user),
):
    try:
        df = _load_df(file)
        new_row = _parse_json_field(data)
        params = _parse_json_field(model_params)

        facade, kind = ModelFactory.create(model_name.value, params)   # ← .value
        if kind != "regression":
            log_and_raise(ModelKindMismatch(details={"expected": "regression", "got": kind}))

        results = facade.train_and_evaluate(df, target_col=target_col)
        pred = facade.predict(pd.DataFrame([new_row]))

        pred_value = float(pred.iloc[0]) if hasattr(pred, "iloc") else (
            float(pred[0]) if isinstance(pred, (list, tuple)) else float(pred)
        )

        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=5,
                action="predict/regression",
                model_name=model_name.value,      # ← .value
                file_name=file.filename,
            )
        except ValueError as e:
            if "not enough tokens" in str(e).lower():
                log_and_raise(TokensNotEnough())
            else:
                log_and_raise(InternalError(details={"cause": str(e)}))

        return {
            "model": model_name.value,           # ← .value
            "target_col": target_col,
            "prediction": pred_value,
            "metrics": {
                "mse": _clean_float(results.get("mse")),
                "rmse": _clean_float(results.get("rmse")),
                "r2": _clean_float(results.get("r2")),
            },
            "used_params": params,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

# ---- Predict by saved model id ----
@app.post("/predict/by_id/{model_id}")
def predict_by_id(
    model_id: str,
    data: str = Form(...),
    user: dict = Depends(get_current_user),
):
    try:
        meta = dal.get_model(model_id)
        if not meta or meta["username"] != user["username"]:
            log_and_raise(ModelNotFound(details={"model_id": model_id}))

        bundle = joblib.load(meta["path"])
        model = bundle.get("model")
        if model is None:
            log_and_raise(ModelBundleCorrupt(details={"path": meta["path"]}))
        feature_cols = bundle.get("feature_cols")
        if not feature_cols:
            log_and_raise(ModelBundleCorrupt(details={"missing": "feature_cols"}))

        try:
            new_row = json.loads(data)
            if not isinstance(new_row, dict):
                raise ValueError("data is not a JSON object")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"data must be a JSON object of features: {e}")

        X_new = pd.DataFrame([new_row]).reindex(columns=feature_cols, fill_value=0)

        scaler = bundle.get("scaler")
        if scaler is not None:
            X_new = pd.DataFrame(scaler.transform(X_new), columns=feature_cols)

        poly = bundle.get("poly")
        if poly is not None:
            X_new = poly.transform(X_new)

        y_hat = model.predict(X_new)

        if meta["kind"] == "classification":
            pred_value = y_hat[0] if hasattr(y_hat, "__len__") else y_hat
            resp = {"prediction": str(pred_value)}
        else:
            val = float(y_hat[0] if hasattr(y_hat, "__len__") else y_hat)
            resp = {"prediction": val}

        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=5,
                action="predict/by_id",
                model_name=meta["model_name"],
                file_name=None,
            )
            logger.info(f"User '{user['username']}' predicted by_id model_id='{model_id}'")
        except ValueError as e:
            raise HTTPException(status_code=402, detail=str(e))

        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---- Predict by latest saved model name ----
@app.post("/predict/{model_name}")
def predict_model_name(
    model_name: ModelName,                    # ← Enum
    data: Dict[str, Any] = Body(...),
    user: dict = Depends(get_current_user),
):
    models = dal.list_models(user["username"])
    meta = next((m for m in models if m["model_name"] == model_name.value), None)  # ← .value
    if not meta:
        raise HTTPException(status_code=404, detail=f"No trained model found for '{model_name.value}'")

    try:
        bundle = joblib.load(meta["path"])
        feature_cols = bundle["feature_cols"]
        scaler = bundle.get("scaler")
        poly = bundle.get("poly")
        model = bundle.get("model")
        if model is None:
            raise HTTPException(status_code=500, detail="Saved model bundle is missing 'model'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model bundle: {e}")

    try:
        X_new = pd.DataFrame([data]).reindex(columns=feature_cols, fill_value=0)
        if scaler is not None:
            X_new = pd.DataFrame(scaler.transform(X_new), columns=feature_cols)
        if poly is not None:
            X_new = poly.transform(X_new)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad input data: {e}")

    try:
        y_hat = model.predict(X_new)
        if meta["kind"] == "classification":
            pred_value = y_hat[0] if hasattr(y_hat, "__len__") else y_hat
            response = {"prediction": str(pred_value)}
        else:
            val = float(y_hat[0] if hasattr(y_hat, "__len__") else y_hat)
            response = {"prediction": val}

        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=5,
                action=f"predict/{model_name.value}",  # ← .value
                model_name=model_name.value,           # ← .value
                file_name=None,
            )
            logger.info(f"User '{user['username']}' predicted with model='{model_name.value}' kind='{meta['kind']}'")
        except ValueError as e:
            logger.warning(f"User '{user['username']}' tried to predict without enough tokens")
            raise HTTPException(status_code=402, detail=str(e))

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- Tokens / Users ----
@app.get("/tokens/{username}")
def get_tokens(username: str, user: dict = Depends(get_current_user)):
    if username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to view other users' tokens")
    u = dal.get_user(username)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return {"tokens": int(u["tokens"])}

@app.post("/add_tokens")
def add_tokens(payload: AddTokens, user: dict = Depends(get_current_user)):
    if payload.username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to add tokens to other users")
    if payload.amount <= 0:
        raise HTTPException(status_code=400, detail="amount must be positive")
    u = dal.get_user(payload.username)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    new_total = int(u["tokens"]) + int(payload.amount)
    dal.update_tokens(payload.username, new_total)
    dal.log_usage(payload.username, "add_tokens", None, None, tokens_after_usage=new_total)
    logger.info(f"User '{payload.username}' added {payload.amount} tokens (new total: {new_total})")
    return {"username": payload.username, "tokens": new_total}

@app.delete("/remove_user")
def remove_user(payload: RemoveUser, user: dict = Depends(get_current_user)):
    if payload.username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to remove other users")
    u = dal.get_user(payload.username)
    if not u or not verify_password(payload.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="Bad credentials")
    with sqlite3.connect(dal.DB_NAME) as conn:
        db = conn.cursor()
        db.execute("DELETE FROM models WHERE username=?", (payload.username,))
        db.execute("DELETE FROM usage_logs WHERE username=?", (payload.username,))
        db.execute("DELETE FROM users WHERE username=?", (payload.username,))
        conn.commit()
        logger.warning(f"User '{payload.username}' removed account")
    return {"message": f"user '{payload.username}' removed"}
