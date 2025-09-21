import os
import io
from typing import List, Tuple, Optional
from pathlib import Path

from google.cloud import storage
from PIL import Image

from config import GCS_BUCKET_NAME, GCS_CREDENTIALS_PATH


class GCSStorage:
    def __init__(self):
        # Initialize GCS client
        if GCS_CREDENTIALS_PATH and os.path.exists(GCS_CREDENTIALS_PATH):
            self.client = storage.Client.from_service_account_json(GCS_CREDENTIALS_PATH)
        else:
            # Use default credentials (for environments like GKE)
            self.client = storage.Client()
        
        self.bucket_name = GCS_BUCKET_NAME
        self.bucket = self.client.bucket(self.bucket_name)

    def save_image(self, pil_image: Image.Image, folder_name: str, filename: str) -> str:
        """Save PIL image to GCS bucket.
        
        Args:
            pil_image: PIL Image object
            folder_name: Folder/prefix in bucket (e.g., 'mazda_cx5', '_negative')
            filename: Image filename (e.g., 'uuid.jpg')
            
        Returns:
            GCS path in format 'folder_name/filename'
        """
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create blob path
        blob_path = f"{folder_name}/{filename}"
        blob = self.bucket.blob(blob_path)
        
        # Upload image
        blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
        
        return blob_path

    def load_image(self, blob_path: str) -> Image.Image:
        """Load image from GCS bucket.
        
        Args:
            blob_path: Full path in bucket (e.g., 'mazda_cx5/uuid.jpg')
            
        Returns:
            PIL Image object
        """
        blob = self.bucket.blob(blob_path)
        image_bytes = blob.download_as_bytes()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def list_gallery(self) -> List[Tuple[str, str, bool]]:
        """List all images in the gallery bucket.
        
        Returns:
            List of tuples: (label, blob_path, is_negative)
        """
        items = []
        
        # List all blobs in the bucket
        blobs = self.client.list_blobs(self.bucket_name)
        
        for blob in blobs:
            # Skip if not an image file
            if not blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
                
            # Extract folder name (label) from blob path
            path_parts = blob.name.split('/')
            if len(path_parts) < 2:
                continue
                
            label = path_parts[0]
            is_negative = label.startswith('_')
            
            items.append((label, blob.name, is_negative))
        
        return sorted(items)

    def delete_image(self, blob_path: str) -> bool:
        """Delete image from GCS bucket.
        
        Args:
            blob_path: Full path in bucket
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_path)
            blob.delete()
            return True
        except Exception:
            return False

    def image_exists(self, blob_path: str) -> bool:
        """Check if image exists in GCS bucket.
        
        Args:
            blob_path: Full path in bucket
            
        Returns:
            True if exists, False otherwise
        """
        blob = self.bucket.blob(blob_path)
        return blob.exists()

    def get_signed_url(self, blob_path: str, expiration_minutes: int = 60) -> str:
        """Get a signed URL for temporary access to an image.
        
        Args:
            blob_path: Full path in bucket
            expiration_minutes: How long the URL should be valid
            
        Returns:
            Signed URL string
        """
        blob = self.bucket.blob(blob_path)
        from datetime import timedelta
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="GET",
        )


# Global instance
gcs_storage: Optional[GCSStorage] = None


def get_gcs_storage() -> GCSStorage:
    """Get or create global GCS storage instance."""
    global gcs_storage
    if gcs_storage is None:
        gcs_storage = GCSStorage()
    return gcs_storage
