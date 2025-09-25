import json
import gzip
import tempfile
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import faiss

from config import (
    INDEX_PATH, LABELS_PATH, NEG_INDEX_PATH, PROTO_PATH, META_PATH,
    PROTOTYPE_MODE, ENABLE_GCS_BACKUP, BACKUP_ON_SINGLE_ADD,
    BACKUP_BATCH_SIZE, BACKUP_INTERVAL_MINUTES, MAX_BACKUP_VERSIONS
)
from ml_models import get_embed_dim

import logging
logger = logging.getLogger(__name__)


# Backup state tracking
_backup_state = {
    "last_backup_time": None,
    "additions_since_backup": 0,
    "pending_backup": False
}


def _should_backup_now(force: bool = False) -> bool:
    """Determine if we should backup now based on configured policies"""
    if force or not ENABLE_GCS_BACKUP:
        return force and ENABLE_GCS_BACKUP

    if BACKUP_ON_SINGLE_ADD:
        return True

    # Check batch size threshold
    if _backup_state["additions_since_backup"] >= BACKUP_BATCH_SIZE:
        return True

    # Check time interval
    if _backup_state["last_backup_time"] is None:
        return True

    time_since_backup = datetime.now() - _backup_state["last_backup_time"]
    if time_since_backup.total_seconds() >= (BACKUP_INTERVAL_MINUTES * 60):
        return True

    return False


def _mark_addition():
    """Mark that an addition occurred (for batch tracking)"""
    _backup_state["additions_since_backup"] += 1
    _backup_state["pending_backup"] = True


def _mark_backup_completed():
    """Mark that a backup was completed"""
    _backup_state["last_backup_time"] = datetime.now()
    _backup_state["additions_since_backup"] = 0
    _backup_state["pending_backup"] = False


def save_meta():
    META_PATH.write_text(json.dumps({
        "embed_dim": get_embed_dim(),
        "prototype_mode": PROTOTYPE_MODE
    }))


def save_pos_index(
    ix: faiss.Index, 
    labels: List[str], 
    backup_to_gcs: bool = None, 
    force_backup: bool = False
):
    # Save locally for fast access
    faiss.write_index(ix, str(INDEX_PATH))
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False))

    # Smart backup logic
    if backup_to_gcs is None:
        should_backup = _should_backup_now(force_backup)
    else:
        should_backup = backup_to_gcs

    if should_backup:
        _backup_index_to_gcs("positive", ix, labels)
        _mark_backup_completed()
    elif not force_backup:
        _mark_addition()


def save_neg_index(
    ix: Optional[faiss.Index], 
    backup_to_gcs: bool = None, 
    force_backup: bool = False
):
    if ix is None:
        if NEG_INDEX_PATH.exists():
            NEG_INDEX_PATH.unlink()
        return

    # Save locally for fast access
    faiss.write_index(ix, str(NEG_INDEX_PATH))

    # Smart backup logic
    if backup_to_gcs is None:
        should_backup = _should_backup_now(force_backup)
    else:
        should_backup = backup_to_gcs

    if should_backup:
        _backup_index_to_gcs("negative", ix, [])
        _mark_backup_completed()
    elif not force_backup:
        _mark_addition()


def save_prototypes(
    proto: Dict[str, Dict[str, List[float] | int]], 
    backup_to_gcs: bool = None
):
    PROTO_PATH.write_text(json.dumps(proto))  # raw sums+counts

    # Smart backup logic for prototypes
    if backup_to_gcs is None:
        backup_to_gcs = ENABLE_GCS_BACKUP
    if backup_to_gcs:
        _backup_prototypes_to_gcs(proto)


def load_indexes():
    """Load persisted indexes. Returns (index_pos, labels_pos, index_neg)"""
    index_pos = None
    labels_pos = []
    index_neg = None

    if INDEX_PATH.exists() and LABELS_PATH.exists():
        index_pos = faiss.read_index(str(INDEX_PATH))
        labels_pos = json.loads(LABELS_PATH.read_text())

    if NEG_INDEX_PATH.exists():
        index_neg = faiss.read_index(str(NEG_INDEX_PATH))

    return index_pos, labels_pos, index_neg


def load_prototypes():
    """Load prototypes if they exist"""
    if PROTO_PATH.exists():
        return json.loads(PROTO_PATH.read_text())
    return {}


def load_prototypes_with_fallback():
    """Load prototypes with GCS fallback if local file doesn't exist"""
    try:
        # Try loading from local storage first (fast)
        if PROTO_PATH.exists():
            return json.loads(PROTO_PATH.read_text())

        # If no local prototypes, try to restore from GCS backup
        logger.info(
            "Local prototypes not found, attempting to restore from GCS backup..."
        )

        prototypes = restore_prototypes_from_gcs()
        if prototypes:
            # Save restored prototypes locally for future fast access
            save_prototypes(prototypes, backup_to_gcs=False)
            logger.info("✅ Successfully restored prototypes from GCS backup")
            return prototypes

        logger.info("No prototypes backup found in GCS")
        return {}

    except Exception as e:
        logger.warning(f"Failed to load prototypes with fallback: {e}")
        return {}


def _backup_prototypes_to_gcs(prototypes: Dict[str, Dict[str, List[float] | int]]):
    """Backup prototypes to GCS with compression and versioning"""
    try:
        from gcs_storage import get_gcs_storage

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Save prototypes to temporary file
            temp_proto_path = temp_dir_path / f"prototypes_{timestamp}.json"
            with open(temp_proto_path, 'w') as f:
                json.dump(prototypes, f, ensure_ascii=False)

            # Compress prototypes
            compressed_proto_path = temp_dir_path / f"prototypes_{timestamp}.json.gz"
            with open(temp_proto_path, 'rb') as f_in:
                with gzip.open(compressed_proto_path, 'wb') as f_out:
                    f_out.writelines(f_in)

            # Upload to GCS
            gcs = get_gcs_storage()
            _upload_file_to_gcs(
                gcs, 
                compressed_proto_path, 
                f"indexes/prototypes_{timestamp}.json.gz"
            )

            # Update latest pointer
            _update_latest_prototypes_pointer(gcs, timestamp)

            # Clean up old prototype backups
            _cleanup_old_prototype_backups(gcs)

    except Exception as e:
        # Log but don't fail on backup errors
        logger.warning(f"Failed to backup prototypes to GCS: {e}")


def _backup_index_to_gcs(index_type: str, ix: faiss.Index, labels: List[str]):
    """Backup FAISS index to GCS with compression and versioning"""
    try:
        from gcs_storage import get_gcs_storage

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Save index to temporary file
            temp_index_path = temp_dir_path / f"{index_type}_index.faiss"
            faiss.write_index(ix, str(temp_index_path))

            # Compress index
            compressed_index_path = (
                temp_dir_path / f"{index_type}_index_{timestamp}.faiss.gz"
            )
            with open(temp_index_path, 'rb') as f_in:
                with gzip.open(compressed_index_path, 'wb') as f_out:
                    f_out.writelines(f_in)

            # Save labels if provided
            if labels:
                labels_path = temp_dir_path / f"{index_type}_labels_{timestamp}.json.gz"
                labels_json = json.dumps(labels, ensure_ascii=False).encode('utf-8')
                with gzip.open(labels_path, 'wb') as f:
                    f.write(labels_json)

            # Upload to GCS
            gcs = get_gcs_storage()

            # Upload compressed index
            _upload_file_to_gcs(
                gcs, 
                compressed_index_path, 
                f"indexes/{index_type}_index_{timestamp}.faiss.gz"
            )

            # Upload labels if provided
            if labels:
                _upload_file_to_gcs(
                    gcs, 
                    labels_path, 
                    f"indexes/{index_type}_labels_{timestamp}.json.gz"
                )

            # Update latest symlink
            _update_latest_index_pointer(gcs, index_type, timestamp)

            # Clean up old backups to keep only MAX_BACKUP_VERSIONS
            _cleanup_old_backups(gcs, index_type)

    except Exception as e:
        # Log but don't fail on backup errors
        logger.warning(f"Failed to backup {index_type} index to GCS: {e}")


def _upload_file_to_gcs(gcs_storage, local_path: Path, gcs_path: str):
    """Upload a file to GCS bucket"""
    blob = gcs_storage.bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


def _update_latest_prototypes_pointer(gcs_storage, timestamp: str):
    """Update pointer to latest prototypes version"""
    pointer_data = {
        "latest_timestamp": timestamp,
        "created_at": datetime.now().isoformat()
    }

    pointer_blob = gcs_storage.bucket.blob("indexes/prototypes_latest.json")
    pointer_blob.upload_from_string(json.dumps(pointer_data))


def _update_latest_index_pointer(gcs_storage, index_type: str, timestamp: str):
    """Update pointer to latest index version"""
    pointer_data = {
        "latest_timestamp": timestamp,
        "created_at": datetime.now().isoformat()
    }

    pointer_blob = gcs_storage.bucket.blob(f"indexes/{index_type}_latest.json")
    pointer_blob.upload_from_string(json.dumps(pointer_data))


def _cleanup_old_backups(gcs_storage, index_type: str):
    """Remove old backup versions, keeping only MAX_BACKUP_VERSIONS latest"""
    try:
        # Get all backup timestamps for this index type
        versions = list_index_versions(index_type)

        if len(versions) <= MAX_BACKUP_VERSIONS:
            return  # Nothing to clean up

        # Calculate how many to delete
        versions_to_delete = versions[MAX_BACKUP_VERSIONS:]  # Keep first N, delete rest
        deleted_count = 0

        for timestamp in versions_to_delete:
            try:
                # Delete index file
                index_blob_path = f"indexes/{index_type}_index_{timestamp}.faiss.gz"
                index_blob = gcs_storage.bucket.blob(index_blob_path)
                if index_blob.exists():
                    index_blob.delete()
                    deleted_count += 1

                # Delete labels file if it exists
                labels_blob_path = f"indexes/{index_type}_labels_{timestamp}.json.gz"
                labels_blob = gcs_storage.bucket.blob(labels_blob_path)
                if labels_blob.exists():
                    labels_blob.delete()

            except Exception as e:
                logger.warning(
                    f"Failed to delete backup {timestamp} for {index_type}: {e}"
                )

        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} old {index_type} index backups, "
                f"keeping {MAX_BACKUP_VERSIONS} latest"
            )

    except Exception as e:
        logger.warning(f"Failed to cleanup old {index_type} backups: {e}")


def _cleanup_old_prototype_backups(gcs_storage):
    """Remove old prototype backup versions, keeping only MAX_BACKUP_VERSIONS latest"""
    try:
        # Get all prototype backup timestamps
        versions = list_prototype_versions()

        if len(versions) <= MAX_BACKUP_VERSIONS:
            return  # Nothing to clean up

        # Calculate how many to delete
        versions_to_delete = versions[MAX_BACKUP_VERSIONS:]  # Keep first N, delete rest
        deleted_count = 0

        for timestamp in versions_to_delete:
            try:
                # Delete prototypes file
                proto_blob_path = f"indexes/prototypes_{timestamp}.json.gz"
                proto_blob = gcs_storage.bucket.blob(proto_blob_path)
                if proto_blob.exists():
                    proto_blob.delete()
                    deleted_count += 1

            except Exception as e:
                logger.warning(f"Failed to delete prototype backup {timestamp}: {e}")

        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} old prototype backups, "
                f"keeping {MAX_BACKUP_VERSIONS} latest"
            )

    except Exception as e:
        logger.warning(f"Failed to cleanup old prototype backups: {e}")


def restore_index_from_gcs(
    index_type: str, 
    timestamp: Optional[str] = None
) -> Optional[tuple]:
    """Restore FAISS index from GCS backup

    Args:
        index_type: "positive" or "negative"
        timestamp: specific version to restore, or None for latest

    Returns:
        Tuple of (index, labels) or None if not found
    """
    try:
        from gcs_storage import get_gcs_storage
        gcs = get_gcs_storage()

        # Get timestamp to restore
        if timestamp is None:
            # Get latest version
            try:
                pointer_blob = gcs.bucket.blob(f"indexes/{index_type}_latest.json")
                pointer_data = json.loads(pointer_blob.download_as_text())
                timestamp = pointer_data["latest_timestamp"]
            except Exception:
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Download and decompress index
            index_blob_path = f"indexes/{index_type}_index_{timestamp}.faiss.gz"
            index_blob = gcs.bucket.blob(index_blob_path)

            if not index_blob.exists():
                return None

            compressed_index_path = temp_dir_path / "index.faiss.gz"
            index_blob.download_to_filename(str(compressed_index_path))

            # Decompress index
            index_path = temp_dir_path / "index.faiss"
            with gzip.open(compressed_index_path, 'rb') as f_in:
                with open(index_path, 'wb') as f_out:
                    f_out.writelines(f_in)

            # Load index
            index = faiss.read_index(str(index_path))

            # Download and decompress labels if they exist
            labels = []
            labels_blob_path = f"indexes/{index_type}_labels_{timestamp}.json.gz"
            labels_blob = gcs.bucket.blob(labels_blob_path)

            if labels_blob.exists():
                compressed_labels_path = temp_dir_path / "labels.json.gz"
                labels_blob.download_to_filename(str(compressed_labels_path))

                with gzip.open(compressed_labels_path, 'rb') as f:
                    labels_json = f.read().decode('utf-8')
                    labels = json.loads(labels_json)

            return index, labels

    except Exception as e:
        logger.warning(f"Failed to restore {index_type} index from GCS: {e}")
        return None


def load_indexes_with_fallback():
    """Load indexes with GCS fallback if local files don't exist"""
    try:
        # Try loading from local storage first (fast)
        return load_indexes()
    except Exception:
        # Fallback to GCS restore
        logger.info("Local indexes not found, attempting GCS restore...")

        pos_result = restore_index_from_gcs("positive")
        neg_result = restore_index_from_gcs("negative")

        index_pos = pos_result[0] if pos_result else None
        labels_pos = pos_result[1] if pos_result else []
        index_neg = neg_result[0] if neg_result else None

        # Save restored indexes locally for future fast access
        if index_pos:
            save_pos_index(index_pos, labels_pos, backup_to_gcs=False)
        if index_neg:
            save_neg_index(index_neg, backup_to_gcs=False)

        return index_pos, labels_pos, index_neg


def restore_prototypes_from_gcs(timestamp: Optional[str] = None) -> Optional[Dict]:
    """Restore prototypes from GCS backup"""
    try:
        from gcs_storage import get_gcs_storage
        gcs = get_gcs_storage()

        # Get timestamp to restore
        if timestamp is None:
            # Get latest version
            try:
                pointer_blob = gcs.bucket.blob("indexes/prototypes_latest.json")
                pointer_data = json.loads(pointer_blob.download_as_text())
                timestamp = pointer_data["latest_timestamp"]
            except Exception:
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Download and decompress prototypes
            proto_blob_path = f"indexes/prototypes_{timestamp}.json.gz"
            proto_blob = gcs.bucket.blob(proto_blob_path)

            if not proto_blob.exists():
                return None

            compressed_proto_path = temp_dir_path / "prototypes.json.gz"
            proto_blob.download_to_filename(str(compressed_proto_path))

            # Decompress prototypes
            proto_path = temp_dir_path / "prototypes.json"
            with gzip.open(compressed_proto_path, 'rb') as f_in:
                with open(proto_path, 'wb') as f_out:
                    f_out.writelines(f_in)

            # Load prototypes
            with open(proto_path, 'r') as f:
                prototypes = json.load(f)

            return prototypes

    except Exception as e:
        logger.warning(f"Failed to restore prototypes from GCS: {e}")
        return None


def list_prototype_versions() -> List[str]:
    """List available prototype versions in GCS"""
    try:
        from gcs_storage import get_gcs_storage
        gcs = get_gcs_storage()

        blobs = gcs.client.list_blobs(gcs.bucket_name, prefix="indexes/prototypes_")
        timestamps = []

        for blob in blobs:
            if blob.name.endswith('.json.gz'):
                # Extract timestamp from filename
                parts = blob.name.split('_')
                if len(parts) >= 2:
                    timestamp_part = parts[-1].replace('.json.gz', '')
                    timestamps.append(timestamp_part)

        return sorted(timestamps, reverse=True)  # Latest first

    except Exception:
        return []


def list_index_versions(index_type: str) -> List[str]:
    """List available index versions in GCS"""
    try:
        from gcs_storage import get_gcs_storage
        gcs = get_gcs_storage()

        blobs = gcs.client.list_blobs(
            gcs.bucket_name, 
            prefix=f"indexes/{index_type}_index_"
        )
        timestamps = []

        for blob in blobs:
            if blob.name.endswith('.faiss.gz'):
                # Extract timestamp from filename
                parts = blob.name.split('_')
                if len(parts) >= 3:
                    timestamp_part = parts[-1].replace('.faiss.gz', '')
                    timestamps.append(timestamp_part)

        return sorted(timestamps, reverse=True)  # Latest first

    except Exception:
        return []


def force_backup_now() -> Dict[str, any]:
    """Force an immediate backup of current indexes if they exist"""
    results = {"positive": False, "negative": False, "prototypes": False, "errors": []}

    try:
        # Backup positive index if it exists
        if INDEX_PATH.exists() and LABELS_PATH.exists():
            index_pos = faiss.read_index(str(INDEX_PATH))
            labels_pos = json.loads(LABELS_PATH.read_text())
            _backup_index_to_gcs("positive", index_pos, labels_pos)
            results["positive"] = True
    except Exception as e:
        results["errors"].append(f"Positive index backup failed: {e}")

    try:
        # Backup negative index if it exists
        if NEG_INDEX_PATH.exists():
            index_neg = faiss.read_index(str(NEG_INDEX_PATH))
            _backup_index_to_gcs("negative", index_neg, [])
            results["negative"] = True
    except Exception as e:
        results["errors"].append(f"Negative index backup failed: {e}")

    try:
        # Backup prototypes if they exist
        if PROTO_PATH.exists():
            prototypes = json.loads(PROTO_PATH.read_text())
            _backup_prototypes_to_gcs(prototypes)
            results["prototypes"] = True
    except Exception as e:
        results["errors"].append(f"Prototypes backup failed: {e}")

    if results["positive"] or results["negative"] or results["prototypes"]:
        _mark_backup_completed()

    return results


def cleanup_old_backups_all() -> Dict[str, any]:
    """Manually cleanup old backups for all index types"""
    results = {
        "positive": {"cleaned": 0, "kept": 0},
        "negative": {"cleaned": 0, "kept": 0},
        "prototypes": {"cleaned": 0, "kept": 0},
        "errors": []
    }

    try:
        from gcs_storage import get_gcs_storage
        gcs = get_gcs_storage()

        # Cleanup index backups
        for index_type in ["positive", "negative"]:
            try:
                versions = list_index_versions(index_type)
                total_versions = len(versions)

                if total_versions <= MAX_BACKUP_VERSIONS:
                    results[index_type]["kept"] = total_versions
                    continue

                versions_to_delete = versions[MAX_BACKUP_VERSIONS:]
                deleted_count = 0

                for timestamp in versions_to_delete:
                    try:
                        # Delete index file
                        index_blob_path = (
                            f"indexes/{index_type}_index_{timestamp}.faiss.gz"
                        )
                        index_blob = gcs.bucket.blob(index_blob_path)
                        if index_blob.exists():
                            index_blob.delete()
                            deleted_count += 1

                        # Delete labels file if it exists
                        labels_blob_path = (
                            f"indexes/{index_type}_labels_{timestamp}.json.gz"
                        )
                        labels_blob = gcs.bucket.blob(labels_blob_path)
                        if labels_blob.exists():
                            labels_blob.delete()

                    except Exception as e:
                        results["errors"].append(
                            f"Failed to delete {index_type} backup {timestamp}: {e}"
                        )

                results[index_type]["cleaned"] = deleted_count
                results[index_type]["kept"] = MAX_BACKUP_VERSIONS

            except Exception as e:
                results["errors"].append(f"Failed to cleanup {index_type} backups: {e}")

        # Cleanup prototype backups
        try:
            versions = list_prototype_versions()
            total_versions = len(versions)

            if total_versions <= MAX_BACKUP_VERSIONS:
                results["prototypes"]["kept"] = total_versions
            else:
                versions_to_delete = versions[MAX_BACKUP_VERSIONS:]
                deleted_count = 0

                for timestamp in versions_to_delete:
                    try:
                        # Delete prototypes file
                        proto_blob_path = f"indexes/prototypes_{timestamp}.json.gz"
                        proto_blob = gcs.bucket.blob(proto_blob_path)
                        if proto_blob.exists():
                            proto_blob.delete()
                            deleted_count += 1

                    except Exception as e:
                        results["errors"].append(
                            f"Failed to delete prototypes backup {timestamp}: {e}"
                        )

                results["prototypes"]["cleaned"] = deleted_count
                results["prototypes"]["kept"] = MAX_BACKUP_VERSIONS

        except Exception as e:
            results["errors"].append(f"Failed to cleanup prototypes backups: {e}")

    except Exception as e:
        results["errors"].append(f"Failed to connect to GCS: {e}")

    return results


def get_backup_status() -> Dict[str, any]:
    """Get current backup status and statistics"""
    # Get version counts for each index type
    backup_versions = {}
    try:
        for index_type in ["positive", "negative"]:
            versions = list_index_versions(index_type)
            backup_versions[index_type] = len(versions)

        # Add prototype versions
        proto_versions = list_prototype_versions()
        backup_versions["prototypes"] = len(proto_versions)
    except Exception:
        backup_versions = {"positive": 0, "negative": 0, "prototypes": 0}

    return {
        "last_backup_time": (
            _backup_state["last_backup_time"].isoformat() 
            if _backup_state["last_backup_time"] else None
        ),
        "additions_since_backup": _backup_state["additions_since_backup"],
        "pending_backup": _backup_state["pending_backup"],
        "backup_versions": backup_versions,
        "backup_settings": {
            "enable_gcs_backup": ENABLE_GCS_BACKUP,
            "backup_on_single_add": BACKUP_ON_SINGLE_ADD,
            "backup_batch_size": BACKUP_BATCH_SIZE,
            "backup_interval_minutes": BACKUP_INTERVAL_MINUTES,
            "max_backup_versions": MAX_BACKUP_VERSIONS
        },
        "next_backup_triggers": {
            "batch_threshold": (
                f"{_backup_state['additions_since_backup']}/{BACKUP_BATCH_SIZE}"
            ),
            "time_threshold": f"Every {BACKUP_INTERVAL_MINUTES} minutes"
        }
    }
