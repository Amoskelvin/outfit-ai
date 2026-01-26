import os
import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException
from src.config.settings import settings

class FileHandler:
    @staticmethod
    def validate_image(file: UploadFile) -> bool:
        """Validate uploaded image file"""
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Check file size (limit to 10MB)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_file_size / 1024 / 1024}MB"
            )
        
        return True
    
    @staticmethod
    async def save_upload(file: UploadFile, user_id: int) -> dict:
        """Save uploaded file and return file info"""
        # Validate file
        FileHandler.validate_image(file)
        
        # Generate unique filename
        file_ext = Path(file.filename).suffix.lower()
        unique_filename = f"{user_id}_{uuid.uuid4()}{file_ext}"
        
        # Create user directory if it doesn't exist
        user_dir = Path(settings.upload_dir) / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = user_dir / unique_filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "filename": unique_filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "original_filename": file.filename
        }