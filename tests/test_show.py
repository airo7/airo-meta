import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("show", Path(__file__).parent.parent / "scripts" / "show.py")
show = importlib.util.module_from_spec(spec)
spec.loader.exec_module(show)

CAR = ["--camera", "http://phone:8080/video", "--floor", "0.80", "1.22", "--marker-h", "0.1", "--dry"]
DEMO = ["python", "scripts/demo.py"]


def test_car_arguments_go_to_the_car_and_no_env_file_is_read_by_default():
    children = show.commands(*show.parse(CAR))
    assert children["warehouse"] == (8001, ["uv", "run", "--no-env-file", *DEMO, "--site", "factory", "--busy",
                                            "normal", "--host", "127.0.0.1", "--port", "8001"])
    assert children["car"] == (8002, ["uv", "run", "--no-env-file", *DEMO, *CAR, "--host", "127.0.0.1",
                                      "--port", "8002"])


def test_env_file_ports_and_switches():
    args = show.parse(["--env-file", ".env", "--no-warehouse", "--car-port", "8022", *CAR])
    assert show.commands(*args) == {"car": (8022, ["uv", "run", "--env-file", ".env", *DEMO, *CAR, "--host",
                                                   "127.0.0.1", "--port", "8022"])}
    calm = show.commands(*show.parse(["--no-car", "--busy", "calm"]))
    assert list(calm) == ["warehouse"] and "calm" in calm["warehouse"][1]
