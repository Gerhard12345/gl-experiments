import re
import toml

version_string = toml.load("pyproject.toml")["project"]["requires-python"]
match = re.search(r"\d+\.\d+", version_string)
print(match.group(0))
