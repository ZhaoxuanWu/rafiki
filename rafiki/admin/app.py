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

from flask import Flask, request, jsonify, g, make_response
from flask_cors import CORS
import os
import traceback
import json

from rafiki.constants import UserType
from rafiki.utils.auth import generate_token, decode_token, auth

from .admin import Admin

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Rafiki Admin is up.'

####################################
# Users
####################################

@app.route('/user', methods=['POST'])
@auth([UserType.ADMIN])
def create_user(auth):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.create_user(**params))

@app.route('/users', methods=['POST'])
@auth([UserType.ADMIN])
def create_users(auth):
    admin = get_admin()
    params = get_request_params()

    # Expect csv file as bytes
    csv_file_bytes = request.files['csv_file_bytes'].read()
    params['csv_file_bytes'] = csv_file_bytes

    with admin:
        return jsonify(admin.create_users(**params))

@app.route('/tokens', methods=['POST'])
def generate_user_token():
    admin = get_admin()
    params = get_request_params()

    # Error will be thrown here if credentials are invalid
    with admin:
        user = admin.authenticate_user(**params)

    auth = {
        'user_id': user['id'],
        'user_type': user['user_type']
    }
    
    token = generate_token(auth)

    return jsonify({
        'user_id': user['id'],
        'user_type': user['user_type'],
        'token': token
    })

####################################
# Train Jobs
####################################

@app.route('/train_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def create_train_job(auth):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.create_train_job(auth['user_id'], **params))

@app.route('/train_jobs', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_train_jobs(auth):
    admin = get_admin()
    params = get_request_params()

    if 'user_id' in params:
        with admin:
            return jsonify(admin.get_train_jobs_by_user(params['user_id']))

@app.route('/train_jobs/<app>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_train_jobs_of_app(auth, app):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_train_jobs_of_app(app, **params))

@app.route('/train_jobs/<app>/<app_version>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_train_job(auth, app, app_version):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_train_job(app, app_version=int(app_version), **params))

@app.route('/train_jobs/<app>/<app_version>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def stop_train_job(auth, app, app_version):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.stop_train_job(app, app_version=int(app_version), **params))

@app.route('/train_jobs/<app>/<app_version>/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trials_of_train_job(auth, app, app_version):
    admin = get_admin()
    params = get_request_params()

    # Return best trials by train job
    if params.get('type') == 'best':
        del params['type']

        if 'max_count' in params:
            params['max_count'] = int(params['max_count'])

        with admin:
            return jsonify(admin.get_best_trials_of_train_job(
                app, 
                app_version=int(app_version),
                **params
            ))
    
    # Return all trials by train job
    else:
        with admin:
            return jsonify(admin.get_trials_of_train_job(
                app, 
                app_version=int(app_version),
                **params)
            )

@app.route('/train_job_workers/<service_id>/stop', methods=['POST'])
@auth([])
def stop_train_job_worker(auth, service_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.stop_train_job_worker(service_id, **params))

####################################
# Trials
####################################

@app.route('/trials/<trial_id>/logs', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial_logs(auth, trial_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_trial_logs(trial_id, **params))

@app.route('/trials/<trial_id>/parameters', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial_parameters(auth, trial_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        parameters = admin.get_trial_parameters(trial_id, **params)

    res = make_response(parameters)
    res.headers.set('Content-Type', 'application/octet-stream')
    return res

@app.route('/trials/<trial_id>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial(auth, trial_id):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_trial(trial_id, **params))

####################################
# Inference Jobs
####################################

@app.route('/inference_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def create_inference_jobs(auth):
    admin = get_admin()
    params = get_request_params()

    if 'app_version' in params:
        params['app_version'] = int(params['app_version'])

    with admin:
        return jsonify(admin.create_inference_job(auth['user_id'], **params))

@app.route('/inference_jobs', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_inference_jobs(auth):
    admin = get_admin()
    params = get_request_params()

    if 'user_id' in params:
        with admin:
            return jsonify(admin.get_inference_jobs_by_user(params['user_id']))

@app.route('/inference_jobs/<app>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_inference_jobs_of_app(auth, app):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_inference_jobs_of_app(app, **params))

@app.route('/inference_jobs/<app>/<app_version>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_running_inference_job(auth, app, app_version):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.get_running_inference_job(app, app_version=int(app_version), **params))

@app.route('/inference_jobs/<app>/<app_version>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def stop_inference_job(auth, app, app_version=-1):
    admin = get_admin()
    params = get_request_params()

    with admin:
        return jsonify(admin.stop_inference_job(app, app_version=int(app_version), **params))

####################################
# Models
####################################

@app.route('/models', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def create_model(auth):
    admin = get_admin()
    params = get_request_params()

    # Expect model file as bytes
    model_file_bytes = request.files['model_file_bytes'].read()
    params['model_file_bytes'] = model_file_bytes

    # Expect model dependencies as dict
    if 'dependencies' in params and isinstance(params['dependencies'], str):
        params['dependencies'] = json.loads(params['dependencies'])

    with admin:
        return jsonify(admin.create_model(auth['user_id'], **params))

@app.route('/models/<name>/model_file', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_model_file(auth, name):
    admin = get_admin()
    params = get_request_params()

    with admin:
        model_file = admin.get_model_file(auth['user_id'], name, **params)

    res = make_response(model_file)
    res.headers.set('Content-Type', 'application/octet-stream')
    return res

@app.route('/models/<name>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_model(auth, name):
    admin = get_admin()
    params = get_request_params()
    with admin:
        return jsonify(admin.get_model(auth['user_id'], name, **params))

@app.route('/models', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_models(auth):
    admin = get_admin()
    params = get_request_params()

    # Return models by task
    if params.get('task') is not None:
        with admin:
            return jsonify(admin.get_models_of_task(auth['user_id'], **params))
    
    # Return all models
    else:
        with admin:
            return jsonify(admin.get_models(auth['user_id'], **params))

# Handle uncaught exceptions with a server error & the error's stack trace (for development)
@app.errorhandler(Exception)
def handle_error(error):
    return traceback.format_exc(), 500

# Extract request params from Flask request
def get_request_params():
    # Get params from body as JSON
    params = request.get_json()

    # If the above fails, get params from body as form data
    if params is None:
        params = request.form.to_dict()

    # Merge in query params
    query_params = {
        k: v
        for k, v in request.args.items()
    }
    params = {**params, **query_params}

    return params

def get_admin():
    # Allow multiple threads to each have their own instance of admin
    if not hasattr(g, 'admin'):
        g.admin = Admin()
    
    return g.admin
