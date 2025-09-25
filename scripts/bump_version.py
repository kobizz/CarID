#!/usr/bin/env python3
"""
Version bumping script for the parking service.
Updates both pyproject.toml and config.yaml with synchronized versions.
"""

import re
import sys
import argparse
from pathlib import Path


def get_current_version():
    """Get current version from pyproject.toml"""
    pyproject_path = Path('pyproject.toml')
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
        match = re.search(r'version = "([^"]+)"', content)
        if not match:
            raise ValueError("Could not find version in pyproject.toml")
        return match.group(1)


def bump_semantic_version(current, bump_type):
    """Bump version according to semantic versioning"""
    try:
        major, minor, patch = map(int, current.split('.'))
    except ValueError:
        raise ValueError(f"Invalid semantic version format: {current}")
    
    if bump_type == 'patch':
        patch += 1
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return f"{major}.{minor}.{patch}"


def update_version_in_file(filename, new_version):
    """Update version in specified file"""
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"{filename} not found")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if filename == 'pyproject.toml':
        content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)
    elif filename == 'config.yaml':
        content = re.sub(r'version: "[^"]+"', f'version: "{new_version}"', content)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    
    with open(filepath, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description='Bump version in pyproject.toml and config.yaml')
    parser.add_argument('version', help='New version (e.g., 1.2.3) or bump type (patch/minor/major)')
    args = parser.parse_args()

    try:
        # Get current version
        current_version = get_current_version()
        print(f"Current version: {current_version}")

        # Determine new version
        if args.version in ['patch', 'minor', 'major']:
            new_version = bump_semantic_version(current_version, args.version)
        else:
            # Validate that it looks like a version
            if not re.match(r'^\d+\.\d+\.\d+$', args.version):
                raise ValueError(f"Invalid version format: {args.version}")
            new_version = args.version

        print(f"New version: {new_version}")

        # Update both files
        update_version_in_file('pyproject.toml', new_version)
        update_version_in_file('config.yaml', new_version)

        print(f"✅ Updated pyproject.toml and config.yaml to version {new_version}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
