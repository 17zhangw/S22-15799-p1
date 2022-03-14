#!/bin/bash

doit project1_reset_db

# Clone Andy's BenchBase.
doit benchbase_clone --repo_url="https://github.com/apavlo/benchbase.git" --branch_name="main"
cp ./build/benchbase/config/postgres/15799_starter_config.xml ./config/behavior/benchbase/epinions_config.xml

# Generate the ePinions config file.
mkdir -p artifacts/project/
cp ./config/behavior/benchbase/epinions_config.xml ./artifacts/project/epinions_config.xml
xmlstarlet edit --inplace --update '/parameters/url' --value "jdbc:postgresql://localhost:5432/project1db?preferQueryMode=simple" ./artifacts/project/epinions_config.xml
xmlstarlet edit --inplace --update '/parameters/username' --value "project1user" ./artifacts/project/epinions_config.xml
xmlstarlet edit --inplace --update '/parameters/password' --value "project1pass" ./artifacts/project/epinions_config.xml
xmlstarlet edit --inplace --update '/parameters/scalefactor' --value "1" ./artifacts/project/epinions_config.xml

# Load ePinions.
doit benchbase_run --benchmark="epinions" --config="./artifacts/project/epinions_config.xml" --args="--create=true --load=true"

# Collect the workload trace of executing ePinions.
doit project1_enable_logging
doit benchbase_run --benchmark="epinions" --config="./artifacts/project/epinions_config.xml" --args="--execute=true"
doit project1_disable_logging

# Look at your collected files (and any old traces -- beware!).
sudo ls -lah /var/lib/postgresql/14/main/log/ | grep csv
