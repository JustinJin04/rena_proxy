#!/bin/bash

category=filesys
port=9000
classifier=/m-coriander/coriander/zhan/DualTune/config/filesys/classifier.json
tool_adapters=/m-coriander/coriander/zhan/DualTune/config/filesys/tool_adapters.json
tool_list=/m-coriander/coriander/zhan/DualTune/config/filesys/tool_list.json

python -m rena_proxy.proxy \
  --category $category \
  --classifier $classifier \
  --tool_adapters $tool_adapters \
  --tool_list $tool_list \
  --port $port
