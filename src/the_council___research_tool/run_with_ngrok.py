#!/usr/bin/env python3
"""
Script to run Streamlit app with ngrok tunnel for external access
"""
import subprocess
import time
import signal
import sys
from pyngrok import ngrok

def main():
    # Set your ngrok auth token if you have one
    # ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

    print("🚀 Starting Streamlit app with ngrok tunnel...")

    # Start ngrok tunnel for port 8501 (default Streamlit port)
    try:
        public_url = ngrok.connect(8501)
        print(f"✅ Ngrok tunnel established!")
        print(f"🌐 Public URL: {public_url}")
        print(f"📱 Local URL: http://localhost:8501")
        print(f"🔗 Share this URL: {public_url}")
        print("\nPress Ctrl+C to stop the tunnel and exit.")

        # Keep the tunnel running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down ngrok tunnel...")
        ngrok.disconnect(public_url)
        ngrok.kill()
        print("✅ Tunnel closed. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure ngrok is properly installed and authenticated.")
        sys.exit(1)

if __name__ == "__main__":
    main()
