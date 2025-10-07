import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

JWT_SECRET_KEY = secrets.token_urlsafe(32) 
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

security = HTTPBearer()


class APIKey(BaseModel):
    key_hash: str
    name: str
    rate_limit_per_minute: int = 60
    total_requests: int = 0
    last_used_at: Optional[datetime] = None
    is_active: bool = True


class TokenData(BaseModel):
    api_key_hash: str
    exp: datetime


API_KEYS_STORE: Dict[str, APIKey] = {}


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key(name: str, rate_limit: int = 60) -> str:
    key = f"sync_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(key)
    
    API_KEYS_STORE[key_hash] = APIKey(
        key_hash=key_hash,
        name=name,
        rate_limit_per_minute=rate_limit,
        is_active=True
    )
    
    return key


def verify_api_key(key: str) -> Optional[APIKey]:
    key_hash = hash_api_key(key)
    api_key = API_KEYS_STORE.get(key_hash)
    
    if not api_key or not api_key.is_active:
        return None
    
    api_key.total_requests += 1
    api_key.last_used_at = datetime.utcnow()
    
    return api_key


def create_jwt_token(api_key_hash: str) -> str:
    expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    payload = {
        "api_key_hash": api_key_hash,
        "exp": expiration,
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def verify_jwt_token(token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenData(
            api_key_hash=payload["api_key_hash"],
            exp=datetime.fromtimestamp(payload["exp"])
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> APIKey:
    token = credentials.credentials
    
    api_key = verify_api_key(token)
    if api_key:
        return api_key
    
    token_data = verify_jwt_token(token)
    if token_data:
        api_key = API_KEYS_STORE.get(token_data.api_key_hash)
        if api_key and api_key.is_active:
            api_key.total_requests += 1
            api_key.last_used_at = datetime.utcnow()
            return api_key
    
    raise HTTPException(
        status_code=401,
        detail="Invalid or expired authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[APIKey]:
    if not credentials:
        return None
    
    try:
        return await get_current_api_key(credentials)
    except HTTPException:
        return None


def init_default_keys():
    admin_key = "sync_admin_key_2024"
    admin_hash = hash_api_key(admin_key)
    API_KEYS_STORE[admin_hash] = APIKey(
        key_hash=admin_hash,
        name="Admin Key",
        rate_limit_per_minute=1000,
        is_active=True
    )
    
    test_key = "sync_test_key_2024"
    test_hash = hash_api_key(test_key)
    API_KEYS_STORE[test_hash] = APIKey(
        key_hash=test_hash,
        name="Test Key",
        rate_limit_per_minute=60,
        is_active=True
    )
    
    print(f"Initialized default API keys:")
    print(f"  Admin: {admin_key}")
    print(f"  Test: {test_key}")


class RateLimiter:    
    def __init__(self):
        self.requests: Dict[str, list] = {}
    
    def check_rate_limit(self, api_key: APIKey) -> bool:
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        key_hash = api_key.key_hash
        if key_hash not in self.requests:
            self.requests[key_hash] = []
        
        self.requests[key_hash] = [
            req_time for req_time in self.requests[key_hash]
            if req_time > minute_ago
        ]
        
        if len(self.requests[key_hash]) >= api_key.rate_limit_per_minute:
            return False
        
        self.requests[key_hash].append(now)
        return True


rate_limiter = RateLimiter()


async def check_rate_limit(api_key: APIKey = Depends(get_current_api_key)):
    if not rate_limiter.check_rate_limit(api_key):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Limit: {api_key.rate_limit_per_minute} requests/minute",
        )
    
    return api_key


init_default_keys()
