import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

# Construct database URL from environment
PG_USER = os.getenv("PG_USER", "maestro")
PG_PASS = os.getenv("PG_PASS", "supersecret")
# In docker-compose, the host is 'postgres'. Locally it might be '127.0.0.1'
PG_HOST = os.getenv("PG_HOST", "127.0.0.1") 
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "maestro_auth")

SQLALCHEMY_DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# Support SQLite for tests
if os.getenv("TESTING", "False").lower() == "true":
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, 
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



