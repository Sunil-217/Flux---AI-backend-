from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.db import get_db
from app.core.security import create_access_token, get_current_user
from app.models import User
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Request bodies ──
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: str


class VerifyRequest(BaseModel):
    email: EmailStr
    code: str


class ResendRequest(BaseModel):
    email: EmailStr


class SigninRequest(BaseModel):
    email: EmailStr
    password: str


def _user_out(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email, "phone": user.phone}


def _token_response(user: User) -> dict:
    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "user": _user_out(user),
    }


@router.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if not req.name.strip():
        raise HTTPException(400, "Name is required.")
    try:
        auth_service.signup(
            db, req.name.strip(), req.email.lower(), req.password, req.phone.strip()
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"message": "Verification code sent to your email.", "email": req.email.lower()}


@router.post("/verify-otp")
def verify_otp(req: VerifyRequest, db: Session = Depends(get_db)):
    try:
        user = auth_service.verify_otp(db, req.email.lower(), req.code.strip())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return _token_response(user)


@router.post("/resend-otp")
def resend_otp(req: ResendRequest, db: Session = Depends(get_db)):
    try:
        auth_service.resend_otp(db, req.email.lower())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"message": "A new verification code was sent to your email."}


@router.post("/signin")
def signin(req: SigninRequest, db: Session = Depends(get_db)):
    try:
        user = auth_service.signin(db, req.email.lower(), req.password)
    except ValueError as exc:
        raise HTTPException(401, str(exc))
    return _token_response(user)


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return _user_out(user)
