#!/usr/bin/env python3
"""
Setup script for Sperm Labelbox Uploader
This script helps users configure their environment and validate their setup.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    if not env_example.exists():
        print("❌ env.example file not found")
        return
    
    # Copy template to .env
    with open(env_example) as f:
        content = f.read()
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Created .env file from template")
    print("📝 Please edit .env with your actual values")

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ['LABELBOX_API_KEY', 'PROJECT_ID']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("💡 Please set these in your .env file or environment")
        return False
    
    print("✅ All required environment variables are set")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['labelbox', 'PIL', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def test_labelbox_connection():
    """Test connection to Labelbox API."""
    try:
        import labelbox as lb
        api_key = os.getenv('LABELBOX_API_KEY')
        if not api_key:
            print("❌ LABELBOX_API_KEY not set")
            return False
        
        client = lb.Client(api_key)
        # Try to get user info to test connection
        user = client.get_user()
        print(f"✅ Connected to Labelbox as: {user.email}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to connect to Labelbox: {e}")
        return False

def main():
    """Run setup and validation."""
    print("🚀 Sperm Labelbox Uploader Setup")
    print("=" * 40)
    
    # Step 1: Create .env file
    print("\n1. Setting up environment file...")
    create_env_file()
    
    # Step 2: Check dependencies
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Step 3: Validate environment
    print("\n3. Validating environment...")
    env_ok = validate_environment()
    
    # Step 4: Test connection (only if env is OK)
    if env_ok:
        print("\n4. Testing Labelbox connection...")
        conn_ok = test_labelbox_connection()
    else:
        conn_ok = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Setup Summary:")
    print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"   Environment:  {'✅' if env_ok else '❌'}")
    print(f"   Connection:   {'✅' if conn_ok else '❌'}")
    
    if deps_ok and env_ok and conn_ok:
        print("\n🎉 Setup complete! You're ready to upload data to Labelbox.")
        print("\n📚 Next steps:")
        print("   1. Prepare your data directory with images and YOLO annotations")
        print("   2. Run: python src/labelbox_uploader.py --help")
    else:
        print("\n⚠️  Setup incomplete. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
