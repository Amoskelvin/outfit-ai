import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import init_db, engine
from src.database.models import Base, User
from sqlalchemy.orm import Session

def setup_database():
    """Initialize database with tables and test data"""
    print("Creating database tables...")
    
    try:
        # Create all tables
        init_db()
        print("✅ Tables created successfully!")
        
        # Create a test user
        print("\nCreating test user...")
        with Session(engine) as session:
            # Check if user already exists
            existing_user = session.query(User).filter_by(username="testuser").first()
            
            if not existing_user:
                test_user = User(
                    username="testuser",
                    email="test@example.com"
                )
                session.add(test_user)
                session.commit()
                print(f"✅ Test user created with ID: {test_user.id}")
            else:
                print(f"ℹ️  Test user already exists with ID: {existing_user.id}")
        
        print("\n🎉 Database setup complete!")
        print("\nYou can now:")
        print("1. Run the API: python run.py")
        print("2. Upload images via http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ Error setting up database: {e}")
        print("\nTroubleshooting:")
        print("1. Check that PostgreSQL is running")
        print("2. Verify your DATABASE_URL in .env file")
        print("3. Ensure the database 'outfit_ai' exists")
        sys.exit(1)

if __name__ == "__main__":
    setup_database()