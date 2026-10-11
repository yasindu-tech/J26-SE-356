import importlib
import sys

import pytest


def test_app_refuses_to_start_with_the_dev_secret_outside_dev(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    for name in [m for m in sys.modules if m == "app.main" or m.startswith("app.main.")]:
        del sys.modules[name]
    import app.core.config as config_module

    config_module.get_settings.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="insecure dev default"):
            importlib.import_module("app.main")
    finally:
        sys.modules.pop("app.main", None)
        config_module.get_settings.cache_clear()
