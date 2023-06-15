#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from huggingface_hub import snapshot_download


repo_id = sys.argv[1]
print(snapshot_download(repo_id=repo_id, allow_patterns=["*.json", "*.pt", "*.bin", "*.txt", "*.model"]))
