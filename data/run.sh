#!/usr/bin/env bash
set -euo pipefail

/bin/bash /app/init.sh
/bin/bash /app/train.sh
/bin/bash /app/test.sh
