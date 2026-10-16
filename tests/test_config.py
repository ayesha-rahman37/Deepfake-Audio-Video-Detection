
from pathlib import Path
import yaml

config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuration loaded successfully!")
print("Project:", config["project"]["name"])
print("Window duration:", config["data"]["window_duration"], "seconds")
print("Frames per window:", config["data"]["frames_per_window"])
