from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use a mock/default URL. In production, this comes from env vars.
SQLALCHEMY_DATABASE_URL = "postgresql://maestro:maestro_password@localhost:5432/maestro"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
