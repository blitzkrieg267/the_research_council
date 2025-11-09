#!/usr/bin/env python3
"""
Script to make Streamlit app accessible externally using ngrok
"""
import subprocess
import time
import sys
from pyngrok import ngrok

def main():
    print("🚀 Setting up external access for Document Summarizer...")

    # Check if ngrok auth token is set
    try:
        # Try to get ngrok tunnels to check if authenticated
        tunnels = ngrok.get_tunnels()
        if not tunnels:
            print("⚠️  No active ngrok tunnels found.")
            print("📝 For better experience, sign up at https://ngrok.com and get an auth token:")
            print("   Then run: ngrok config add-authtoken YOUR_TOKEN")
            print("   Or uncomment and set the token in this script.\n")
    except:
        print("⚠️  Ngrok not authenticated. Limited to 2-hour sessions.")
        print("📝 Get a free account at https://ngrok.com for unlimited access.\n")

    print("📋 Instructions:")
    print("1. Open a new terminal window")
    print("2. Run: cd src/the_council___research_tool && python3 -m streamlit run streamlit_app.py")
    print("3. Wait for Streamlit to start (you'll see the local URL)")
    print("4. Come back to this terminal and run: python3 run_with_ngrok.py")
    print("5. Share the public URL that appears!")

    print("\n" + "="*60)
    print("🎯 Quick Start Commands:")
    print("Terminal 1: cd src/the_council___research_tool && python3 -m streamlit run streamlit_app.py")
    print("Terminal 2: cd src/the_council___research_tool && python3 run_with_ngrok.py")
    print("="*60)

    # Offer to start ngrok now
    response = input("\n❓ Start ngrok tunnel now? (y/n): ").lower().strip()
    if response == 'y':
        print("\n🔄 Starting ngrok tunnel...")
        try:
            public_url = ngrok.connect(8501)
            print(f"✅ Success! Public URL: {public_url}")
            print(f"🔗 Share this link: {public_url}")
            print("\n🛑 Press Ctrl+C to stop the tunnel")

            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            ngrok.kill()
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Make sure Streamlit is running on port 8501 first!")

if __name__ == "__main__":
    main()
