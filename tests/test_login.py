import os
import sys
from dotenv import load_dotenv
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
load_dotenv()

import embedding_adapters.cli as cli


def test_cli_login_calls_login(monkeypatch):
    """
    Ensure that running `embedding-adapters login` calls auth.login()
    without raising SystemExit.
    """
    # Save original argv
    original_argv = sys.argv.copy()

    try:
        # Simulate: embedding-adapters login
        sys.argv = ["embedding-adapters", "login"]

        with patch("embedding_adapters.cli.login") as mock_login:
            cli.main()
            mock_login.assert_called_once()
    finally:
        # Restore argv so we don't break other tests
        sys.argv = original_argv