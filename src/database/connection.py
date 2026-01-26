from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config.settings import settings
from src.database.models import Base

# Create engine
engine = create_engine(
    settings.database_url,
    echo=settings.debug  # Log SQL queries in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")