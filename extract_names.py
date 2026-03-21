from pathlib import Path
import yaml

directory = Path("data/paths")  # change this if needed

names = []

for path in directory.glob("*.y*ml"):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for key in ("start", "end"):
        section = data.get(key, {})
        name = section.get("name")
        if isinstance(name, str):
            names.append(name)

deduped_names = list(dict.fromkeys(names))

for name in deduped_names:
    print(name)
