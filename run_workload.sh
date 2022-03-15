#!/bin/bash

# Collect the workload trace of executing ePinions.
doit project1_enable_logging
doit benchbase_run --benchmark="epinions" --config="./artifacts/project/epinions_config.xml" --args="--execute=true"
doit project1_disable_logging

# Look at your collected files (and any old traces -- beware!).
sudo ls -lah /var/lib/postgresql/14/main/log/ | grep csv
