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

import time
import logging
import os
import uuid
import traceback
import pprint

from .advisor import Advisor

logger = logging.getLogger(__name__)

class InvalidAdvisorException(Exception): pass
class InvalidProposalException(Exception): pass

class AdvisorService(object):
    def __init__(self):
        self._advisors = {}

    def create_advisor(self, knob_config, advisor_id=None):
        is_created = False
        advisor = None

        if advisor_id is not None:
            advisor = self._get_advisor(advisor_id)
            
        if advisor is None:
            advisor = Advisor(knob_config)
            advisor_id = str(uuid.uuid4()) if advisor_id is None else advisor_id
            self._advisors[advisor_id] = advisor
            is_created = True

        return {
            'id': advisor_id,
            'is_created': is_created # Whether a new advisor has been created
        }

    def delete_advisor(self, advisor_id):
        is_deleted = False

        advisor = self._get_advisor(advisor_id)

        if advisor is not None:
            del self._advisors[advisor_id]
            is_deleted = True

        return {
            'id': advisor_id,
            # Whether the advisor has been deleted (maybe it already has been deleted)
            'is_deleted': is_deleted 
        }

    def generate_proposal(self, advisor_id):
        advisor = self._get_advisor(advisor_id)
        knobs = advisor.propose()

        return {
            'knobs': knobs
        }

    # Feedbacks to the advisor on the score of a set of knobs
    # Additionally, returns another proposal of knobs after ingesting feedback
    def feedback(self, advisor_id, knobs, score):
        advisor = self._get_advisor(advisor_id)

        if advisor is None:
            raise InvalidAdvisorException()

        advisor.feedback(knobs, score)
        knobs = advisor.propose()

        return {
            'knobs': knobs
        }

    def _get_advisor(self, advisor_id):
        if advisor_id not in self._advisors:
            return None

        advisor = self._advisors[advisor_id]
        return advisor
