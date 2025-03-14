#!/usr/bin/env python3
"""
This script fixes SSL certificate verification issues on macOS.
"""
import os
import sys
import site
from pathlib import Path

def install_certificates():
    """Install certificates for macOS Python."""
    print("Installing certificates for macOS Python...")
    
    try:
        import certifi
        print(f"Using certifi from: {certifi.__file__}")
        
        # Get the certifi certificate path
        cert_path = certifi.where()
        print(f"Certificate path: {cert_path}")
        
        # Set environment variables
        print("\nAdd these lines to your ~/.bash_profile or ~/.zshrc:")
        print(f"export SSL_CERT_FILE={cert_path}")
        print(f"export REQUESTS_CA_BUNDLE={cert_path}")
        
        # Create or update pip.conf with certificate configuration
        pip_conf_dir = Path(site.USER_CONFIG_DIR) / "pip"
        pip_conf_dir.mkdir(parents=True, exist_ok=True)
        
        pip_conf_path = pip_conf_dir / "pip.conf"
        
        pip_conf_content = f"""[global]
cert = {cert_path}
"""
        
        with open(pip_conf_path, "w") as f:
            f.write(pip_conf_content)
        
        print(f"\nCreated pip configuration at {pip_conf_path}")
        
        print("\nCertificate installation complete!")
        print("\nYou may need to restart your terminal or shell for the changes to take effect.")
        
        return True
    
    except Exception as e:
        print(f"Error installing certificates: {e}")
        return False

if __name__ == "__main__":
    if sys.platform != "darwin":
        print("This script is only for macOS.")
        sys.exit(1)
    
    install_certificates()