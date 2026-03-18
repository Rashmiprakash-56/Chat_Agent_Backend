import uuid 
from typing import Optional
from fastapi import Depends , Request
from fastapi_users import BaseUserManager,FastAPIUsers,UUIDIDMixin,models
from fastapi_users.authentication import(
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy
)

from fastapi_users.db import SQLAlchemyUserDatabase
from app.utils.database import User,get_user_db
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

SECRET = settings.SECRET_KEY

class UserManager(UUIDIDMixin,BaseUserManager[User,uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user, request:Optional[Request] = None):
        log.info(" User registered: %s", user.id)
        return await super().on_after_register(user, request)

    async def on_after_forgot_password(self, user, token, request:Optional[Request] = None):
        log.info(" Password reset requested for user: %s", user.id)
        return await super().on_after_forgot_password(user, token, request)
    
    async def on_after_request_verify(self, user, token, request:Optional[Request] = None):
        log.info(" Verification requested for user: %s", user.id)
        return await super().on_after_request_verify(user, token, request)
    

async def get_user_manager(user_db : SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

def get_jwt_strategy():
    return JWTStrategy(secret=SECRET,lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name= 'jwt',
    transport= bearer_transport,
    get_strategy= get_jwt_strategy
)

fastapi_users = FastAPIUsers[User,uuid.UUID](get_user_manager,auth_backends=[auth_backend])
current_active_user  = fastapi_users.current_user(active=True)
