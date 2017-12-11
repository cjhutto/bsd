#!/bin/bash
set -e
pwd
cd opt
pwd
pip install .
pytest test
