from src.polish import Polish

from examples.toy_example import solve_toy_example


def test_solve_toy_example(monkeypatch):
    """Test the execution of the toy example without any exceptions"""

    def my_config_logging(self):
        return

    monkeypatch.setattr(Polish, "_config_logging", my_config_logging)
    solve_toy_example()
