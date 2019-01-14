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

import os
from rafiki.utils.service import run_service
from rafiki.db import Database
from rafiki.constants import ServiceType

# Run install command
install_command = os.environ.get('WORKER_INSTALL_COMMAND', '')
exit_code = os.system(install_command)
if exit_code != 0: 
    raise Exception('Install command gave non-zero exit code: "{}"'.format(install_command))

worker = None

def start_service(service_id, service_type):
    global worker

    if service_type == ServiceType.TRAIN:
        from rafiki.worker import TrainWorker
        worker = TrainWorker(service_id)
        worker.start()
    elif service_type == ServiceType.INFERENCE:
        from rafiki.worker import InferenceWorker
        worker = InferenceWorker(service_id)
        worker.start()
    else:
        raise Exception('Invalid service type: {}'.format(service_type))

def end_service(service_id, service_type):
    global worker
    if worker is not None:
        worker.stop()    

db = Database()
run_service(db, start_service, end_service)
