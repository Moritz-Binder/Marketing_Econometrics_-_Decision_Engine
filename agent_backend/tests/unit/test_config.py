import pytest
from src.core.config import get_settings, Settings

def test_settings_instantiation_from_env(tmp_path, monkeypatch):
    """
    Test that settings are correctly loaded from a .env file.
    """
    # 1. Clear env vars to ensure Pydantic doesn't read them instead of our file
    # (Environment variables always take precedence over .env files)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PROJECT_NAME", raising=False)
    
    # 2. Create a temporary .env file with dummy values
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROJECT_NAME=\"Custom Test Project\"\n"
        "GEMINI_API_KEY=\"secret-from-env\"\n"
    )
    
    # 3. Instantiate Settings, pointing it to our temporary .env file
    settings = Settings(_env_file=str(env_file))
    
    # 4. Verify it parsed the values correctly
    assert settings.PROJECT_NAME == "Custom Test Project"
    assert settings.GEMINI_API_KEY == "secret-from-env"