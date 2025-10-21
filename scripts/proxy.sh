#!/bin/bash

category=filesys
port=9015
classifier=/home/zhan/rena/rena_proxy/config/filesys/classifier.json
tool_adapters=/home/zhan/rena/rena_proxy/config/filesys/tool_adaptor.json
tool_list=/home/zhan/rena/rena_proxy/config/filesys/tool_list.json

python -m rena_proxy.proxy \
  --category $category \
  --classifier $classifier \
  --tool_adapters $tool_adapters \
  --tool_list $tool_list \
  --port $port
