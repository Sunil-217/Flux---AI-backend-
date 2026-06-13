from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.db import get_db
from app.core.security import create_access_token, get_current_user
from app.core.rate_limit import limiter
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


class ForgotRequest(BaseModel):
    email: EmailStr


class ResetRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: str


def _user_out(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email, "phone": user.phone}


def _token_response(user: User) -> dict:
    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "user": _user_out(user),
    }


@router.post("/signup")
@limiter.limit("5/minute")
def signup(request: Request, req: SignupRequest, db: Session = Depends(get_db)):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
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
@limiter.limit("10/minute")
def verify_otp(request: Request, req: VerifyRequest, db: Session = Depends(get_db)):
    try:
        user = auth_service.verify_otp(db, req.email.lower(), req.code.strip())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return _token_response(user)


@router.post("/resend-otp")
@limiter.limit("3/minute")
def resend_otp(request: Request, req: ResendRequest, db: Session = Depends(get_db)):
    try:
        auth_service.resend_otp(db, req.email.lower())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"message": "A new verification code was sent to your email."}


@router.post("/signin")
@limiter.limit("10/minute")
def signin(request: Request, req: SigninRequest, db: Session = Depends(get_db)):
    try:
        user = auth_service.signin(db, req.email.lower(), req.password)
    except ValueError as exc:
        raise HTTPException(401, str(exc))
    return _token_response(user)


@router.post("/forgot-password")
@limiter.limit("3/minute")
def forgot_password(request: Request, req: ForgotRequest, db: Session = Depends(get_db)):
    # Stay generic AND never 500 (a raw 500 strips CORS headers → the browser
    # shows a confusing "CORS blocked" instead of a real error). Email-send
    # failures are swallowed here; the user just retries.
    try:
        auth_service.request_password_reset(db, req.email.lower())
    except Exception:
        pass
    # Always generic — never reveal whether the email is registered.
    return {"message": "If an account exists for this email, a reset code has been sent."}


@router.post("/reset-password")
@limiter.limit("10/minute")
def reset_password(request: Request, req: ResetRequest, db: Session = Depends(get_db)):
    if len(req.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    try:
        user = auth_service.reset_password(
            db, req.email.lower(), req.code.strip(), req.new_password
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return _token_response(user)


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return _user_out(user)


class UpdateProfileRequest(BaseModel):
    name: str
    phone: str | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/profile")
def update_profile(
    req: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        updated = auth_service.update_profile(db, user, req.name, req.phone)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return _user_out(updated)


@router.post("/change-password")
def change_password(
    req: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        auth_service.change_password(db, user, req.current_password, req.new_password)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"message": "Password updated successfully."}
