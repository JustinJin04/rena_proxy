import json

with open("tool_cap.json", "r") as f:
    orig_tool_cap = json.load(f)

tool_capabilities = {}

for tool_cap in orig_tool_cap["tool_cap"]:
    name = tool_cap["function"]["name"]
    extracted_capabilities = tool_cap["function"]["extracted_capabilities"]
    derived_capabilities = tool_cap["function"]["derived_capabilities"]
    tool_capabilities[name] = {
        "extracted_capabilities": extracted_capabilities,
        "derived_capabilities": derived_capabilities
    }

with open("tool-capabilities-filesys.json", "w") as f:
    json.dump(tool_capabilities, f, indent=2)
