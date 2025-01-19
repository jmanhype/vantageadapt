#!/usr/bin/env python3
"""Version checking script for the project."""

import re
import sys
import subprocess
from typing import Optional, Tuple

def get_current_tag() -> Optional[str]:
    """Get the current git tag if it exists."""
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return tag
    except subprocess.CalledProcessError:
        return None

def validate_version_format(version: str) -> Tuple[bool, str]:
    """
    Validate that the version follows semantic versioning.
    
    Args:
        version: Version string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    pattern = r'^v[0-9]+\.[0-9]+\.[0-9]+(-((alpha|beta|rc)\.[0-9]+)|)$'
    if not re.match(pattern, version):
        return False, f"Version {version} does not match format v1.2.3 or v1.2.3-beta.1"
    return True, ""

def check_changelog(version: str) -> Tuple[bool, str]:
    """
    Check if version is documented in CHANGELOG.md.
    
    Args:
        version: Version to check for
        
    Returns:
        Tuple of (is_documented, error_message)
    """
    try:
        with open("CHANGELOG.md", "r") as f:
            content = f.read()
            if f"[{version}]" not in content:
                return False, f"Version {version} not found in CHANGELOG.md"
            return True, ""
    except FileNotFoundError:
        return False, "CHANGELOG.md not found"

def main() -> int:
    """Main function to check version formatting and documentation."""
    current_tag = get_current_tag()
    if not current_tag:
        print("No tag found on current commit")
        return 1
    
    is_valid, error = validate_version_format(current_tag)
    if not is_valid:
        print(error)
        return 1
    
    is_documented, error = check_changelog(current_tag)
    if not is_documented:
        print(error)
        return 1
    
    print(f"Version {current_tag} is valid and documented")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 