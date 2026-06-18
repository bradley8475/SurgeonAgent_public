from cad.parsing.parse_xml import parse_xml
from cad.api import simulate_pattern

tree = parse_xml("assets/clo_xml/update-Tshirt.xml")

change_json = tree.compile()
print(change_json)

with open("workspace/output.json", "w") as f:
    import json

    json.dump(change_json, f, indent=2)
    f.write("\n")

simulate_pattern(tree, "workspace/simulation")
