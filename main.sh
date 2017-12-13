#!/bin/bash
set -e
pwd
cd opt
pwd
pip install .
cd test
pytest .
