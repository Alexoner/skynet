#!/bin/sh

# both directories will do?
#DIR=$(dirname "$0")
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Adding Python Path: ${DIR}"
PYTHONPATH=$PYTHONPATH:${DIR} jupyter-notebook 1>log 2>&1
