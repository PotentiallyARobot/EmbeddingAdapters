# embedding_adapters/cli.py
import sys
from .auth import login

def main():
    if len(sys.argv) < 2:
        print("Usage: embedding-adapters <command>")
        print("Commands:")
        print("  login   Log in with your API key")
        raise SystemExit(1)

    cmd = sys.argv[1]
    if cmd == "login":
        login()
    else:
        print(f"Unknown command: {cmd}")
        raise SystemExit(1)
