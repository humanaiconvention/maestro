import os
from typing import TypedDict
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

ALGORITHM = "HS256"

def _get_jwt_secret() -> str:
    secret = os.environ.get("MAESTRO_JWT_SECRET")
    if not secret:
        raise RuntimeError("MAESTRO_JWT_SECRET environment variable is not set.")
    return secret

class AuthContext(TypedDict):
    tenant_id: str
    user_id: str

security = HTTPBearer()

async def verify_auth(credentials: HTTPAuthorizationCredentials = Security(security)) -> AuthContext:
    """
    Verifies the JWT token from the Authorization header using HS256.

    Expected JWT Claims:
    - sub: The unique identifier for the user (user_id).
    - tenant_id: The unique identifier for the tenant.

    Algorithm: HS256 (HMAC with SHA-256)
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[ALGORITHM])

        user_id: str = payload.get("sub")
        tenant_id: str = payload.get("tenant_id")
        
        if not user_id or not tenant_id:
            raise HTTPException(
                status_code=401,
                detail="Token is missing required claims: 'sub' (user_id) or 'tenant_id'"
            )

        return {
            "tenant_id": str(tenant_id),
            "user_id": str(user_id)
        }
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials (invalid JWT)"
        )
