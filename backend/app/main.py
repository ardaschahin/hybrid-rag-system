from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, EmailStr
import httpx
import os
from typing import Any
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError
from passlib.context import CryptContext

app = FastAPI(title="AICI Backend API", version="0.2.1")

# --- CORS (needed for browser-based frontend) ---
# Vite dev: http://localhost:5173
# Docker frontend (if you use nginx): http://localhost:8080
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENT_URL = os.getenv("AGENT_URL", "http://127.0.0.1:8001")

# --- JWT settings ---
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "240"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# --- In-memory stores (demo) ---
# email -> {user_id, email, password_hash}
USERS_BY_EMAIL: dict[str, dict[str, Any]] = {}
# user_id -> object_list
SESSION_BY_USER: dict[str, list[dict]] = {}


# ----- Request/Response Models -----
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ObjectsUpdateRequest(BaseModel):
    object_list: list[dict]


class QARequest(BaseModel):
    question: str
    top_k: int | None = 3


# ----- Helpers -----
def _hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def _verify_password(pw: str, pw_hash: str) -> bool:
    return pwd_context.verify(pw, pw_hash)


def _create_access_token(*, sub: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,  # user_id
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXPIRE_MIN)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        sub = payload.get("sub")
        if not sub or not isinstance(sub, str):
            raise HTTPException(status_code=401, detail="Invalid token")
        return sub
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ----- Routes -----
@app.get("/health")
def health():
    return {"status": "ok", "service": "backend"}


@app.post("/auth/register")
def register(req: RegisterRequest):
    email = (req.email or "").strip().lower()
    if email in USERS_BY_EMAIL:
        raise HTTPException(status_code=400, detail="Email already registered")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password too short (min 6)")

    user_id = f"user_{len(USERS_BY_EMAIL) + 1}"
    USERS_BY_EMAIL[email] = {
        "user_id": user_id,
        "email": email,
        "password_hash": _hash_password(req.password),
    }
    SESSION_BY_USER[user_id] = []
    return {"message": "registered", "user_id": user_id, "email": email}


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    email = (req.email or "").strip().lower()
    u = USERS_BY_EMAIL.get(email)
    if not u or not _verify_password(req.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_access_token(sub=u["user_id"])
    return TokenResponse(access_token=token)


@app.get("/session/objects")
def get_objects(user_id: str = Depends(_get_current_user_id)):
    obj_list = SESSION_BY_USER.get(user_id, [])
    return {"object_list": obj_list, "object_count": len(obj_list)}


@app.put("/session/objects")
def update_objects(req: ObjectsUpdateRequest, user_id: str = Depends(_get_current_user_id)):
    if not isinstance(req.object_list, list):
        raise HTTPException(status_code=400, detail="object_list must be a list")

    SESSION_BY_USER[user_id] = req.object_list
    return {"message": "objects_updated", "object_count": len(req.object_list)}


@app.post("/qa")
async def qa(req: QARequest, user_id: str = Depends(_get_current_user_id)):
    object_list = SESSION_BY_USER.get(user_id)
    if object_list is None or object_list == []:
        raise HTTPException(status_code=400, detail="No object_list set. Call PUT /session/objects first.")

    payload = {
        "question": req.question,
        "object_list": object_list,
        "top_k": req.top_k,
        "quote_mode": True,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{AGENT_URL}/answer", json=payload)
        r.raise_for_status()
        return r.json()
