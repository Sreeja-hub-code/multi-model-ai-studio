"""
Run from the project root:
    python run.py
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

if __name__ == "__main__":
    print("\n🚀 AI Studio starting at http://localhost:5000\n")
    # SSE long-lived connections are sensitive to reloader interference.
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)

