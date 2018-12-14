import os
import logging
import traceback
import bcrypt
import pickle
import json

from rafiki.db import Database
from rafiki.constants import ServiceStatus, UserType, ServiceType, TrainJobStatus, TaskType, ModelAccessRights
from rafiki.config import MIN_SERVICE_PORT, MAX_SERVICE_PORT, SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD
from rafiki.container import DockerSwarmContainerManager 
from rafiki.utils.log import JobLogger
from rafiki.utils.graph import build_dag

from .services_manager import ServicesManager

logger = logging.getLogger(__name__)

class UserExistsException(Exception): pass
class InvalidUserException(Exception): pass
class InvalidPasswordException(Exception): pass
class InvalidRunningInferenceJobException(Exception): pass
class InvalidTrainJobException(Exception): pass
class InvalidTrialException(Exception): pass
class RunningInferenceJobExistsException(Exception): pass
class NoModelsForTaskException(Exception): pass

class Admin(object):
    def __init__(self, db=Database(), container_manager=DockerSwarmContainerManager()):
        self._base_worker_image = '{}:{}'.format(os.environ['RAFIKI_IMAGE_WORKER'],
                                                os.environ['RAFIKI_VERSION'])

        self._db = db
        self._services_manager = ServicesManager(db, container_manager)
        
        with self._db:
            self._seed_users()

    ####################################
    # Users
    ####################################

    def authenticate_user(self, email, password):
        user = self._db.get_user_by_email(email)

        if not user: 
            raise InvalidUserException()
        
        if not self._if_hash_matches_password(password, user.password_hash):
            raise InvalidPasswordException()

        return {
            'id': user.id,
            'user_type': user.user_type
        }

    def create_user(self, email, password, user_type):
        user = self._create_user(email, password, user_type)
        return {
            'id': user.id
        }

    ####################################
    # Train Job
    ####################################

    def create_train_job(self, user_id, app,
        task, train_dataset_uri, test_dataset_uri,
        models, ensemble):

        # Compute auto-incremented app version
        train_jobs = self._db.get_train_jobs_of_app(app)
        app_version = max([x.app_version for x in train_jobs], default=0) + 1  

        # Check models are defined
        if len(models) == 0:
            raise NoModelsForTaskException()

        # Check models are associated to task
        registered_models = self._db.get_selected_models_of_task(
            [model['name'] for model in models], 
            task
        )
        if len(registered_models) != len(models):
            raise NoModelsForTaskException()

        # Check ensemble model is registered
        registered_ensemble_model = None
        if ensemble is not None:
            registered_ensemble_model = self._db.get_model_of_task(
                ensemble['name'], 
                TaskType.TASK_ENSEMBLE_MAPPING[task]
            )
            if registered_ensemble_model is None:
                raise NoModelsForTaskException
            else:
                models.append(ensemble)
                registered_models.append(registered_ensemble_model)

        train_job = self._db.create_train_job(
            user_id=user_id,
            app=app,
            app_version=app_version,
            task=task,
            train_dataset_uri=train_dataset_uri,
            test_dataset_uri=test_dataset_uri
        )
        self._db.commit()

        sub_train_jobs = []
        for registered_model in registered_models:
            model = [model for model in models if model['name'] == registered_model.name][0]
            sub_train_job = self._db.create_sub_train_job(
                train_job_id=train_job.id,
                model_id=registered_model.id,
                budget_type=model['budget_type'],
                budget_amount=model['budget_amount']
            )
            self._db.commit()
            sub_train_jobs.append(sub_train_job)

        # Build graph
        graph = build_dag(sub_train_jobs, registered_ensemble_model)
        train_job.graph = graph
        self._db.commit()
        self._services_manager.create_train_services(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }

    def get_train_job_status(self, app, app_version=-1):
        #TODO
        pass

    def stop_train_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobException()

        self._services_manager.stop_train_services(train_job.id)

        return {
            'id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }
            
    def get_train_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobException()

        workers = self._db.get_workers_of_train_job(train_job.id)
        services = [self._db.get_service(x.service_id) for x in workers]
        worker_models = [self._db.get_model(x.model_id) for x in workers]

        return {
            'id': train_job.id,
            'status': train_job.status,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'task': train_job.task,
            'train_dataset_uri': train_job.train_dataset_uri,
            'test_dataset_uri': train_job.test_dataset_uri,
            'datetime_started': train_job.datetime_started,
            'datetime_completed': train_job.datetime_completed,
            'budget_type': train_job.budget_type,
            'budget_amount': train_job.budget_amount,
            'workers': [
                {
                    'service_id': service.id,
                    'status': service.status,
                    'replicas': service.replicas,
                    'datetime_started': service.datetime_started,
                    'datetime_stopped': service.datetime_stopped,
                    'model_name': model.name
                }
                for (worker, service, model) 
                in zip(workers, services, worker_models)
            ]
        }

    def get_train_jobs_of_app(self, app):
        train_jobs = self._db.get_train_jobs_of_app(app)
        return [
            {
                'id': x.id,
                'status': x.status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_uri': x.train_dataset_uri,
                'test_dataset_uri': x.test_dataset_uri,
                'datetime_started': x.datetime_started,
                'datetime_completed': x.datetime_completed,
                'budget_type': x.budget_type,
                'budget_amount': x.budget_amount
            }
            for x in train_jobs
        ]

    def get_best_trials_of_train_job(self, app, app_version=-1, max_count=3):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobException()

        best_trials = self._db.get_best_trials_of_train_job(train_job.id, max_count=max_count)
        best_trials_models = [self._db.get_model(x.model_id) for x in best_trials]
        return [
            {
                'id': trial.id,
                'knobs': trial.knobs,
                'datetime_started': trial.datetime_started,
                'datetime_stopped': trial.datetime_stopped,
                'model_name': model.name,
                'score': trial.score
            }
            for (trial, model) in zip(best_trials, best_trials_models)
        ]

    def get_train_jobs_by_user(self, user_id):
        train_jobs = self._db.get_train_jobs_by_user(user_id)
        return [
            {
                'id': x.id,
                'status': x.status,
                'app': x.app,
                'app_version': x.app_version,
                'task': x.task,
                'train_dataset_uri': x.train_dataset_uri,
                'test_dataset_uri': x.test_dataset_uri,
                'datetime_started': x.datetime_started,
                'datetime_completed': x.datetime_completed,
                'budget_type': x.budget_type,
                'budget_amount': x.budget_amount
            }
            for x in train_jobs
        ]

    def get_trials_of_train_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobException()

        trials = self._db.get_trials_of_train_job(train_job.id)
        trials_models = [self._db.get_model(x.model_id) for x in trials]
        return [
            {
                'id': trial.id,
                'knobs': trial.knobs,
                'datetime_started': trial.datetime_started,
                'status': trial.status,
                'datetime_stopped': trial.datetime_stopped,
                'model_name': model.name,
                'score': trial.score
            }
            for (trial, model) in zip(trials, trials_models)
        ]

    def stop_train_job_worker(self, service_id):
        worker = self._services_manager.stop_train_job_worker(service_id)
        return {
            'service_id': worker.service_id,
            'train_job_id': worker.train_job_id,
            'sub_train_job_id': worker.sub_train_job_id
        }

    ####################################
    # Trials
    ####################################
    
    def get_trial_logs(self, trial_id):
        trial = self._db.get_trial(trial_id)
        if trial is None:
            raise InvalidTrialException()

        job_logger = JobLogger()
        job_logger.import_logs(trial.logs)
        (plots, metrics, messages) = job_logger.read_logs()
        job_logger.destroy()
        
        return {
            'plots': plots,
            'metrics': metrics,
            'messages': messages
        }

    ####################################
    # Inference Job
    ####################################

    def create_inference_job(self, user_id, app, app_version):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidTrainJobException('Have you started a train job for this app?')

        if train_job.status != TrainJobStatus.COMPLETED:
            raise InvalidTrainJobException('Train job has not completed.')

        # Ensure only 1 running inference job for 1 train job
        inference_job = self._db.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is not None:
            raise RunningInferenceJobExistsException()

        inference_job = self._db.create_inference_job(
            user_id=user_id,
            train_job_id=train_job.id
        )
        self._db.commit()

        (inference_job, predictor_service) = \
            self._services_manager.create_inference_services(inference_job.id)

        return {
            'id': inference_job.id,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'predictor_host': self._get_service_host(predictor_service)
        }

    def stop_inference_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidRunningInferenceJobException()

        inference_job = self._db.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is None:
            raise InvalidRunningInferenceJobException()

        inference_job = self._services_manager.stop_inference_services(inference_job.id)
        return {
            'id': inference_job.id,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version
        }

    def get_running_inference_job(self, app, app_version=-1):
        train_job = self._db.get_train_job_by_app_version(app, app_version=app_version)
        if train_job is None:
            raise InvalidRunningInferenceJobException()

        inference_job = self._db.get_running_inference_job_by_train_job(train_job.id)
        if inference_job is None:
            raise InvalidRunningInferenceJobException()
            
        workers = self._db.get_workers_of_inference_job(inference_job.id)
        services = [self._db.get_service(x.service_id) for x in workers]
        predictor_service = self._db.get_service(inference_job.predictor_service_id)
        predictor_host = self._get_service_host(predictor_service)
        worker_trials = [self._db.get_trial(x.trial_id) for x in workers]
        worker_trial_models = [self._db.get_model(x.model_id) for x in worker_trials]

        return {
            'id': inference_job.id,
            'status': inference_job.status,
            'train_job_id': train_job.id,
            'app': train_job.app,
            'app_version': train_job.app_version,
            'datetime_started': inference_job.datetime_started,
            'datetime_stopped': inference_job.datetime_stopped,
            'predictor_host': predictor_host,
            'workers': [
                {
                    'service_id': service.id,
                    'status': service.status,
                    'replicas': service.replicas,
                    'datetime_started': service.datetime_started,
                    'datetime_stopped': service.datetime_stopped,
                    'trial': {
                        'id': trial.id,
                        'score': trial.score,
                        'knobs': trial.knobs,
                        'model_name': model.name
                    }
                }
                for (worker, service, trial, model) 
                in zip(workers, services, worker_trials, worker_trial_models)
            ]
        }

    def get_inference_jobs_of_app(self, app):
        inference_jobs = self._db.get_inference_jobs_of_app(app)
        train_jobs = [self._db.get_train_job(x.train_job_id) for x in inference_jobs]
        predictor_services = [self._db.get_service(x.predictor_service_id) for x in inference_jobs]
        predictor_hosts = [self._get_service_host(x) for x in predictor_services]
        return [
            {
                'id': inference_job.id,
                'status': inference_job.status,
                'train_job_id': train_job.id,
                'app': train_job.app,
                'app_version': train_job.app_version,
                'datetime_started': inference_job.datetime_started,
                'datetime_stopped': inference_job.datetime_stopped,
                'predictor_host': predictor_host
            }
            for (inference_job, train_job, predictor_host) in zip(inference_jobs, train_jobs, predictor_hosts)
        ]

    def get_inference_jobs_by_user(self, user_id):
        inference_jobs = self._db.get_inference_jobs_by_user(user_id)
        train_jobs = [self._db.get_train_job(x.train_job_id) for x in inference_jobs]
        predictor_services = [self._db.get_service(x.predictor_service_id) for x in inference_jobs]
        predictor_hosts = [self._get_service_host(x) for x in predictor_services]
        return [
            {
                'id': inference_job.id,
                'status': inference_job.status,
                'train_job_id': train_job.id,
                'app': train_job.app,
                'app_version': train_job.app_version,
                'datetime_started': inference_job.datetime_started,
                'datetime_stopped': inference_job.datetime_stopped,
                'predictor_host': predictor_host
            }
            for (inference_job, train_job, predictor_host) in zip(inference_jobs, train_jobs, predictor_hosts)
        ]

    ####################################
    # Models
    ####################################

    def create_model(self, user_id, name, task, 
                    model_file_bytes, model_class, docker_image=None, access_rights=ModelAccessRights.PUBLIC):
        model = self._db.create_model(
            user_id=user_id,
            name=name,
            task=task,
            model_file_bytes=model_file_bytes,
            model_class=model_class,
            docker_image=(docker_image or self._base_worker_image),
            access_rights=access_rights
        )

        return {
            'name': model.name 
        }

    def get_models(self):
        models = self._db.get_models()
        return [
            {
                'name': model.name,
                'task': model.task,
                'model_class': model.model_class,
                'datetime_created': model.datetime_created,
                'user_id': model.user_id,
                'docker_image': model.docker_image
            }
            for model in models
        ]

    def get_models_of_task(self, task):
        models = self._db.get_models_of_task(task)
        return [
            {
                'name': model.name,
                'task': model.task,
                'model_class': model.model_class,
                'datetime_created': model.datetime_created,
                'user_id': model.user_id,
                'docker_image': model.docker_image
            }
            for model in models
        ]
        
    ####################################
    # Private / Users
    ####################################

    def _seed_users(self):
        logger.info('Seeding users...')

        # Seed superadmin
        try:
            self._create_user(
                email=SUPERADMIN_EMAIL,
                password=SUPERADMIN_PASSWORD,
                user_type=UserType.SUPERADMIN
            )
        except UserExistsException:
            logger.info('Skipping superadmin creation as it already exists...')

    def _hash_password(self, password):
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return password_hash

    def _if_hash_matches_password(self, password, password_hash):
        return bcrypt.checkpw(password.encode('utf-8'), password_hash)

    def _create_user(self, email, password, user_type):
        password_hash = self._hash_password(password)
        user = self._db.get_user_by_email(email)

        if user is not None:
            raise UserExistsException()

        user = self._db.create_user(email, password_hash, user_type)
        self._db.commit()
        return user

    ####################################
    # Private / Services
    ####################################

    def _get_service_host(self, service):
        return '{}:{}'.format(service.ext_hostname, service.ext_port)

    ####################################
    # Private / Others
    ####################################

    def __enter__(self):
        self.connect()

    def connect(self):
        self._db.connect()

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

    def disconnect(self):
        self._db.disconnect()
        