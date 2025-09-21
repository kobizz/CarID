import json
import base64
import os
from pathlib import Path
from typing import Optional


def load_addon_config() -> dict:
    """Load Home Assistant addon configuration from options.json"""
    options_path = Path("/data/options.json")
    if options_path.exists():
        try:
            config = json.loads(options_path.read_text())
            print(f"Loaded addon config with {len(config)} options")
            # Hide the actual service account JSON
            safe_config = {k: ("***HIDDEN***" if k == "gcs_service_account_json" and v else v)
                          for k, v in config.items()}
            print(f"Addon config: {safe_config}")
            return config
        except Exception as e:
            print(f"Error loading addon config: {e}")
    else:
        print(f"Addon config file not found at {options_path}")
    return {}


def get_gcs_credentials_from_addon() -> Optional[str]:
    """Get GCS credentials and create temporary credentials file if needed"""
    addon_config = load_addon_config()

    # Option 1: Check for file-based credentials first
    config_file_path = Path("/config/carid/service-account.json")
    if config_file_path.exists():
        print(f"Using GCS credentials from {config_file_path}")
        return str(config_file_path)

    # Option 2: Get base64-encoded service account JSON from addon config
    gcs_json_b64 = addon_config.get("gcs_service_account_json", "")

    if not gcs_json_b64:
        print("No gcs_service_account_json found in addon config")
        return None

    print(f"Found base64 service account JSON in addon config (length: {len(gcs_json_b64)} chars)")

    try:
        gcs_json_str = base64.b64decode(gcs_json_b64).decode('utf-8')

        json.loads(gcs_json_str)

        credentials_path = "/tmp/gcs-service-account.json"
        with open(credentials_path, 'w') as f:
            f.write(gcs_json_str)

        os.chmod(credentials_path, 0o600)

        print(f"Using GCS credentials from addon configuration (base64) -> {credentials_path}")
        return credentials_path

    except Exception as e:
        print(f"Error processing GCS service account JSON: {e}")
        return None


def get_addon_gcs_config() -> dict:
    """Get GCS configuration from Home Assistant addon"""
    addon_config = load_addon_config()

    if not addon_config:
        return {}

    config = {}

    # Get GCS settings from addon options
    if "gcs_bucket_name" in addon_config:
        config["GCS_BUCKET_NAME"] = addon_config["gcs_bucket_name"]

    # Handle service account credentials
    credentials_path = get_gcs_credentials_from_addon()
    if credentials_path:
        config["GCS_CREDENTIALS_PATH"] = credentials_path

    return config


def get_addon_service_config() -> dict:
    """Get all service configuration from Home Assistant addon"""
    addon_config = load_addon_config()

    if not addon_config:
        return {}

    config = {}

    # Map addon config to environment variables
    if "accept_threshold" in addon_config:
        config["ACCEPT_THRESHOLD"] = str(addon_config["accept_threshold"])

    if "margin_threshold" in addon_config:
        config["MARGIN_THRESHOLD"] = str(addon_config["margin_threshold"])

    if "neg_accept_cap" in addon_config:
        config["NEG_ACCEPT_CAP"] = str(addon_config["neg_accept_cap"])

    if "prototype_mode" in addon_config:
        config["PROTOTYPE_MODE"] = str(addon_config["prototype_mode"]).lower()

    # Add GCS configuration
    config.update(get_addon_gcs_config())

    return config
