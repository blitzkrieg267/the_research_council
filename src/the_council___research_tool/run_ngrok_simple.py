#!/usr/bin/env python3
"""
Simple script to run ngrok tunnel for Streamlit
"""
import subprocess
import sys
import os

def main():
    print("🚀 Starting ngrok tunnel for Streamlit...")

    try:
        # Try local installation first (our downloaded binary)
        ngrok_path = os.path.expanduser("~/bin/ngrok")
        if os.path.exists(ngrok_path):
            ngrok_cmd = ngrok_path
        else:
            # Try to find ngrok in PATH
            result = subprocess.run(['which', 'ngrok'], capture_output=True, text=True)
            if result.returncode == 0:
                # Check if it's the real ngrok, not pyngrok wrapper
                version_result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
                if 'pyngrok' not in version_result.stderr:
                    ngrok_cmd = 'ngrok'
                else:
                    print("❌ Found pyngrok wrapper, but need real ngrok binary.")
                    print("💡 Please install ngrok:")
                    print("   brew install ngrok")
                    print("   or download from: https://ngrok.com/download")
                    sys.exit(1)
            else:
                print("❌ ngrok not found. Please install it:")
                print("   brew install ngrok")
                print("   or download from: https://ngrok.com/download")
                sys.exit(1)

        print("✅ ngrok found!")
        print("🔗 Starting tunnel on port 8501...")
        print("📝 Press Ctrl+C to stop the tunnel")
        print()

        # Start ngrok tunnel
        subprocess.run([ngrok_cmd, 'http', '8501'])

    except KeyboardInterrupt:
        print("\n🛑 Tunnel stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
