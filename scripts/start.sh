#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# Read from shell configuration file
source ./.env.sh

LOG_FILEPATH=$PWD/logs/start.log
FILE_DIR=$(dirname "$0")

# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

ensure_stable()
{
    echo "Waiting for 10s for $1 to stablize..."
    sleep 10
    if ps -p $! > /dev/null
    then
        echo "$1 is running"
    else
        echo "Error running $1"
        echo "Check the logs at $LOG_FILEPATH"
        exit 1
    fi
}

# Create Docker swarm for Rafiki

title "Creating Docker swarm for Rafiki..."
bash $FILE_DIR/create_docker_swarm.sh

# Pull images from Docker Hub

title "Pulling images for Rafiki from Docker Hub..."
bash $FILE_DIR/pull_images.sh

# Start whole Rafiki stack

title "Starting Rafiki's DB..."
(bash $FILE_DIR/start_db.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's DB"

title "Starting Rafiki's Cache..."
(bash $FILE_DIR/start_cache.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Cache"

title "Starting Rafiki's Admin..."
(bash $FILE_DIR/start_admin.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Admin"

title "Starting Rafiki's Advisor..."
(bash $FILE_DIR/start_advisor.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Advisor"

title "Starting Rafiki's Admin Web..."
(bash $FILE_DIR/start_admin_web.sh &> $LOG_FILEPATH) &
ensure_stable "Rafiki's Admin Web"

echo "To use Rafiki, use Rafiki Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/docs/src/user/quickstart.html"
echo "To configure Rafiki, refer to Rafiki's developer docs at https://nginyc.github.io/rafiki/docs/latest/docs/src/dev/setup.html"
