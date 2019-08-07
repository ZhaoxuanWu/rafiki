from __future__ import absolute_import, division, print_function

import os
import io
import sys
import time

import tensorflow as tf
from tensorflow.train import MomentumOptimizer
from tensorflow.python.ops import control_flow_ops

from tensorflow.python.tools import freeze_graph
from tensorflow.python.client import device_lib
from tensorflow.python.framework.ops import Tensor, Operation
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from functools import partial
import scipy.io.wavfile as wave
import itertools
import tempfile
import base64
import math
import librosa
import six
import collections

from examples.models.speech_recognition.utils.text import Alphabet, levenshtein, text_to_char_array
from ds_ctcdecoder import ctc_beam_search_decoder_batch, ctc_beam_search_decoder, Scorer

from rafiki.model import BaseModel, FixedKnob, IntegerKnob, FloatKnob, CategoricalKnob, \
    dataset_utils, logger, test_model_class, InvalidModelParamsException
from rafiki.constants import TaskType, ModelDependency


class ConfigSingleton:
    _config = None

    def __getattr__(self, name):
        if not ConfigSingleton._config:
            raise RuntimeError("Global configuration not yet initialized.")
        if not hasattr(ConfigSingleton._config, name):
            raise RuntimeError("Configuration option {} not found in config.".format(name))
        return ConfigSingleton._config[name]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


FLAGS = tf.app.flags.FLAGS


class Speech2TextDataLayer:
    def __init__(self, model, params, Config, num_workers, worker_id):
        self._model = model
        self._num_workers = num_workers
        self._worker_id = worker_id
        self.params = params
        self.Config = Config

        self.target_pad_value = 0

        self._files = dataset_utils.load_dataset_of_audio_files(self.params['dataset_uri'], self.params['dataset_dir']).df

        if self.params['mode'] != 'infer':
            cols = ['wav_filename', 'transcript']
        else:
            cols = 'wav_filename'

        self.all_files = self._files.loc[:, cols].values
        self._files = self.split_data(self.all_files)
        self._size = len(self._files)

        """
        num_fft (int): size of fft window to use if features require fft, defaults to smallest power of 2 larger than window size
        window_size (float): size of analysis window in milli-seconds.
        window_stride (float): stride of analysis window in milli-seconds.
        """
        self.params['min_duration'] = -1.0
        self.params['max_duration'] = -1.0
        self.params['window_size'] = 20e-3
        self.params['window_stride'] = 10e-3

        num_fft = 2**math.ceil(math.log2(self.params['window_size'] * FLAGS.audio_sample_rate))
        self.params['num_fft'] = num_fft
        mel_basis = librosa.filters.mel(
            FLAGS.audio_sample_rate,
            num_fft,
            n_mels=self.Config.num_audio_features,
            fmin=0,
            fmax=int(FLAGS.audio_sample_rate/2)
        )
        self.params['mel_basis'] = mel_basis

    def build_graph(self):
        with tf.device('/cpu:0'):
            """Builds data processing graph using ``tf.data`` API"""
            if self.params['mode'] != 'infer':
                self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
                if self.params['mode'] == 'train':
                    self._dataset.shuffle(self._size)
                    self.params['max_duration'] = self.Config.train_max_duration
                self._dataset = self._dataset.repeat()
                self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)
                self._dataset = self._dataset.map(
                    lambda line: tf.py_func(
                        self._parse_audio_transcript_element,
                        [line],
                        [self.Config.dtype, tf.int32, tf.int32, tf.int32, tf.float32],
                        stateful=False,
                    ),
                    num_parallel_calls=8,
                )
                if self.params['max_duration'] > 0:
                    self._dataset = self._dataset.filter(
                        lambda x, x_len, y, y_len, duration:
                        tf.less_equal(duration, self.params['max_duration'])
                    )
                self._dataset = self._dataset.map(
                    lambda x, x_len, y, y_len, duration:
                    [x, x_len, y, y_len],
                    num_parallel_calls=8,
                )
                self._dataset = self._dataset.padded_batch(
                    self.Config.batch_size,
                    padded_shapes=([None, self.Config.num_audio_features], 1, [None], 1),
                    padding_values=(tf.cast(0, self.Config.dtype), 0, self.target_pad_value, 0),
                )
            else:
                indices = self.split_data(
                    np.array(list(map(str, range(len(self.all_files)))))
                )
                self._dataset = tf.data.Dataset.from_tensor_slices(
                    np.hstack((indices[:, np.newaxis], self._files[:, np.newaxis]))
                )
                self._dataset = self._dataset.repeat()
                self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)
                self._dataset = self._dataset.map(
                    lambda line: tf.py_func(
                        self._parse_audio_element,
                        [line],
                        [self.Config.dtype, tf.int32, tf.int32, tf.float32],
                        stateful=False,
                    ),
                    num_parallel_calls=8
                )
                if self.params['max_duration'] > 0:
                    self._dataset = self._dataset.filter(
                        lambda x, x_len, y, y_len, duration:
                        tf.less_equal(duration, self.params['max_duration'])
                    )
                self._dataset =self._dataset.map(
                    lambda x, x_len, idx, duration:
                    [x, x_len, idx],
                    num_parallel_calls=16,
                )
                self._dataset = self._dataset.padded_batch(
                    self.Config.batch_size,
                    padded_shapes=([None, self.Config.num_audio_features], 1, 1)
                )
            print(self._dataset)

            self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE).make_initializable_iterator()

            if self.params['mode'] != 'infer':
                x, x_length, y, y_length = self._iterator.get_next()
                # Need to explicitly set batch size dimension
                y.set_shape([self.Config.batch_size, None])
                y_length = tf.reshape(y_length, [self.Config.batch_size])
            else:
                x, x_length, x_id = self._iterator.get_next()
                x_id = tf.reshape(x_id, self.Config.batch_size)

            x.set_shape([self.Config.batch_size, None, self.Config.num_audio_features])
            x_length = tf.reshape(x_length, [self.Config.batch_size])

            pad_to = self.Config.pad_to
            if pad_to > 0:
                # We do padding with TF for librosa backend
                num_pad = tf.mod(pad_to - tf.mod(tf.reduce_max(x_length), pad_to), pad_to)
                x = tf.pad(x, [[0, 0], [0, num_pad], [0, 0]])

            self._input_tensors = {}
            self._input_tensors["source_tensors"] = [x, x_length]
            if self.params['mode'] != 'infer':
                self._input_tensors['target_tensors'] = [y, y_length]
            else:
                self._input_tensors['target_tensors'] = [x_id]

    def _parse_audio_transcript_element(self, element):
        """Parses tf.data element into audio and text
        Args:
          element: tf.data element
        Returns:
          tuple: source audio features as ``np.array``, length of source sequence,
          target text as `np.array` of ids, target text length.
        """
        audio_filename, transcript = element
        if not six.PY2:
            transcript = str(transcript, 'utf-8')
            audio_filename = str(audio_filename, 'utf-8')
        target_indices = [self.Config.char2idx[c] for c in transcript]
        target = np.array(target_indices)

        source, audio_duration = self.get_speech_features_from_file(audio_filename)

        return source.astype(self.Config.dtype.as_numpy_dtype()),\
               np.int32([len(source)]), \
               np.int32(target), \
               np.int32([len(target)]), \
               np.float32([audio_duration])

    def _parse_audio_element(self, id_and_audio_filename):
        """Parses audio from file and returns array of audio features.
        Args:
          id_and_audio_filename: tuple of sample id and corresponding audio file name.
        Returns:
          tuple: source audio features as ``np.array``, length of source sequence, sample id.
        """
        idx, audio_filename = id_and_audio_filename
        source, audio_duration = self.get_speech_features_from_file(audio_filename)
        return source.astype(self.Config.dtype.as_numpy_dtype()), \
               np.int32([len(source)]), np.int32([idx]), \
               np.float32([audio_duration])

    def get_speech_features_from_file(self, filename):
        """Function to get a numpy array of features, from an audio file.
        Args:
          filename (string): WAVE filename.
        Returns:
          np.array: np.array of audio features with shape=[num_time_steps, num_features].
        """
        sample_freq, signal = wave.read(filename)
        features, duration = self.get_speech_features(signal, sample_freq)

        return features, duration

    def get_speech_features(self, signal, sample_freq):
        """
        Get speech features using either librosa
        Args:
          signal (np.array): np.array containing raw audio signal
          sample_freq (float): sample rate of the signal
        Returns:
          np.array: np.array of audio features with shape=[num_time_steps, num_features].
          audio_duration (float): duration of the signal in seconds
        """

        features_type = self.Config.input_type
        augmentation = self.Config.augmentation
        window_size = self.params['window_size']
        window_stride = self.params['window_stride']
        window_fn = np.hanning
        """
        dither (float): weight of Gaussian noise to apply to input signal for dithering/preventing quantization noise
        norm_per_feature (bool): if True, the output features will be normalized (whitened) individually. 
        """
        dither = 1e-5
        num_fft = self.params['num_fft']
        norm_per_feature = True
        mel_basis = self.params['mel_basis']
        if mel_basis is not None and sample_freq != self.Config.audio_sample_rate:
            raise ValueError(
                ("The sampling frequency set in params {} does not match the "
                 "frequency {} read from file").format(self.Config.audio_sample_rate, sample_freq)
            )
        features, duration = self.get_speech_features_librosa(
            signal, sample_freq, features_type, window_size, window_stride, augmentation,
            window_fn=window_fn, dither=dither, norm_per_feature=norm_per_feature,
            num_fft=num_fft, mel_basis=mel_basis
        )

        return features, duration

    def get_speech_features_librosa(self, signal, sample_freq,
                                    features_type='spectrogram',
                                    window_size=20e-3,
                                    window_stride=10e-3,
                                    augmentation=None,
                                    window_fn=np.hanning,
                                    num_fft=None,
                                    dither=0.0,
                                    norm_per_feature=False,
                                    mel_basis=None):
        """Function to convert raw audio signal to numpy array of features.
        Backend: librosa
        Args:
          signal (np.array): np.array containing raw audio signal.
          sample_freq (float): frames per second.

        Returns:
          np.array: np.array of audio features with shape=[num_time_steps, num_features].
          audio_duration (float): duration of the signal in seconds
        """

        signal = self.normalize_signal(signal.astype(np.float32))
        audio_duration = len(signal) * 1.0 / sample_freq
        num_fft = num_fft or 2 ** math.ceil(math.log2(window_size * sample_freq))

        if dither > 0:
            signal += dither * np.random.randn(*signal.shape)

        if features_type == 'logfbank':
            signal = self.preemphasis(signal, coeff=0.97)
            S = np.abs(librosa.core.stft(signal, n_fft=num_fft,
                                         hop_length=int(window_stride * sample_freq),
                                         win_length=int(window_size * sample_freq),
                                         center=True, window=window_fn)) ** 2.0
            features = np.log(np.dot(mel_basis, S) + 1e-20).T
        else:
            raise ValueError('Unknown features type: {}'.format(features_type))

        norm_axis = 0 if norm_per_feature else None
        mean = np.mean(features, axis=norm_axis)
        std_dev = np.std(features, axis=norm_axis)
        features = (features - mean) / std_dev

        if augmentation:
            n_freq_mask = augmentation.get('n_freq_mask', 0)
            n_time_mask = augmentation.get('n_time_mask', 0)
            width_freq_mask = augmentation.get('width_freq_mask', 10)
            width_time_mask = augmentation.get('width_time_mask', 50)

            for idx in range(n_freq_mask):
                freq_band = np.random.randint(width_freq_mask + 1)
                freq_base = np.random.randint(0, features.shape[1] - freq_band)
                features[:, freq_base:freq_base + freq_band] = 0
            for idx in range(n_time_mask):
                time_band = np.random.randint(width_time_mask + 1)
                if features.shape[0] - time_band > 0:
                    time_base = np.random.randint(features.shape[0] - time_band)
                    features[time_base:time_base + time_band, :] = 0
        return features, audio_duration

    @staticmethod
    def normalize_signal(signal):
        """
        Normalize float32 signal to [-1, 1] range
        """
        return signal / (np.max(np.abs(signal)) + 1e-5)

    @staticmethod
    def preemphasis(signal, coeff=0.97):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    def split_data(self, data):
        if self.params['mode'] != 'train' and self._num_workers is not None:
            size = len(data)
            start = size // self._num_workers * self._worker_id
            if self._worker_id == self._num_workers - 1:
                end = size
            else:
                end = size // self._num_workers * (self._worker_id + 1)
            return data[start:end]
        else:
            return data

    def get_size_in_samples(self):
        """Returns the number of audio files."""
        return len(self._files)

    @property
    def input_tensors(self):
        """Dictionary with input tensors.
        ``input_tensors["source_tensors"]`` contains:
          * source_sequence
            (shape=[batch_size x sequence length x num_audio_features])
          * source_length (shape=[batch_size])
        ``input_tensors["target_tensors"]`` contains:
          * target_sequence
            (shape=[batch_size x sequence length])
          * target_length (shape=[batch_size])
        """
        return self._input_tensors

    @property
    def iterator(self):
        """Underlying tf.data iterator."""
        return self._iterator


class Speech2Text:
    def __init__(self, Config, params):
        self.params = params
        self.Config = Config
        print(FLAGS.print_loss_steps)
        print(Config.available_devices)

        self._encoder = self._create_encoder()
        print("ENCODER!!!!!!!!!!!!", self.encoder)
        self._decoder = self._create_decoder()
        print("DECODER!!!!!!!!!!!!", self.decoder)
        if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
            self._loss_computator = CTCLoss()
        else:
            self._loss_computator = None

        self._data_layers = []
        for worker_id in range(len(self.Config.available_devices)):
            self._data_layers.append(
                Speech2TextDataLayer(
                    model=self,
                    params=params,
                    Config=Config,
                    num_workers=len(self.Config.available_devices),
                    worker_id=worker_id
                )
            )

        if self.params['mode'] == 'train':
            # doing a few less steps if data size is not divisible by the batch size
            self._steps_in_epoch = self.get_data_layer().get_size_in_samples() // self.Config.batch_size
            self._steps_in_epoch //= len(self.Config.available_devices)
            self._last_step = Config.batch_size * self._steps_in_epoch

        self._outputs = [None] * len(self.Config.available_devices)

    def compile(self, force_var_reuse=False):
        """TensorFlow graph is built here"""

        ###################################################################
        initializer = None # Whether to use None

        # Below we follow data parallelism for multi-GPU training
        # self._gpu_ids = [d[-1] for d in self.Config.available_devices]
        losses = []
        for gpu_cnt, gpu_id in enumerate(self.Config.available_devices):
            print('gpu_cnt', gpu_cnt, type(gpu_cnt))
            print('gpu_id', gpu_id, type(gpu_id))
            with tf.device(gpu_id), tf.variable_scope(
                name_or_scope=tf.get_variable_scope(),
                # re-using variables across GPUs
                reuse=force_var_reuse or (gpu_cnt > 0),
                initializer=initializer,
                dtype=tf.float16
            ):
                logger.log("Building graph on {}".format(gpu_id))
                self.get_data_layer(gpu_cnt).build_graph()
                input_tensors = self.get_data_layer(gpu_cnt).input_tensors

                # Build TF graph
                loss, self._outputs[gpu_cnt] = self._build_forward_pass_graph(
                    input_tensors,
                    gpu_id=0
                )

                if self._outputs[gpu_cnt] is not None and not isinstance(self._outputs[gpu_cnt], list):
                    raise ValueError('Decoder outputs have to be either None or list')
                if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
                    losses.append(loss)

        # End of for gpu ind loop
        if self.params['mode'] == 'train':
            self.loss = tf.reduce_mean(losses)
        if self.params['mode'] == 'eval':
            self.eval_losses = losses

        self._num_objects_per_step = [self._get_num_objects_per_step(worker_id)
                                      for worker_id in range(len(self.Config.available_devices))]

        if self.params['mode'] == 'train':
            lr_policy = self._get_learning_rate_policy()

        var_list = tf.trainable_variables()
        freeze_variable_regex = None
        self.train_op = self.optimize_loss(
            loss=tf.cast(self.loss, tf.float32) + self.get_regularization_loss(),
            dtype=self.Config.dtype,
            optimizer=NovoGrad,
            optimizer_params=self.Config.optimizer_params,
            var_list=var_list,
            learning_rate_decay_fn=lr_policy,
            summaries=self.Config.summaries,
            larc_params=self.Config.larc_params,
            loss_scaling=self.Config.loss_scaling,
            iter_size=1,
            model=self
        )

        tf.summary.scalar(name='train_loss', tensor=self.loss)
        if self.steps_in_epoch:
            tf.summary.scalar(
                name="epoch",
                tensor=tf.floor(tf.train.get_global_step() /
                                tf.constant(self.steps_in_epoch, dtype=tf.int64)),
            )

        logger.log("Trainable variables:")
        total_params = 0
        unknown_shape = False
        for var in var_list:
            var_params = 1
            logger.log('{}'.format(var.name), offset=2)
            logger.log('shape: {}, {}'.format(var.get_shape(), var.dtype),
                       offset=4)
            if var.get_shape():
                for dim in var.get_shape():
                    var_params *= dim.value
                total_params += var_params
            else:
                unknown_shape = True
        if unknown_shape:
            logger.log("Encountered unknown variable shape, can't compute total "
                       "number of parameters.")
        else:
            logger.log('Total trainable parameters: {}'.format(total_params))

    def _create_encoder(self):
        return TDNNEncoder(params=self.params, Config=self.Config, model=self)

    def _create_decoder(self):
        self.dump_outputs = self.Config.decoder_params.get('infer_logits_to_pickle', False)

        def sparse_tensor_to_chars(tensor, idx2char):
            text = [''] * tensor.dense_shape[0]
            for idx_tuple, value in zip(tensor.indices, tensor.values):
                text[idx_tuple[0]] += idx2char[value]
            return text

        self.tensor_to_chars = sparse_tensor_to_chars
        self.tensor_to_char_params = {}

        return FullyConnectedCTCDecoder(params=self.params, Config=self.Config, model=self)

    def _get_learning_rate_policy(self):
        def poly_decay(global_step, learning_rate, decay_steps, power=1.0,
                       begin_decay_at=0, min_lr=0.0, warmup_steps=0):
            """Polynomial decay learning rate policy.
            This function is equivalent to ``tensorflow.train.polynomial_decay`` with
            some additional functionality. Namely, it adds ``begin_decay_at`` parameter
            which is the first step to start decaying learning rate.

            Args:
              global_step: global step TensorFlow tensor.
              learning_rate (float): initial learning rate to use.
              decay_steps (int): number of steps to apply decay for.
              power (float): power for polynomial decay.
              begin_decay_at (int): the first step to start decaying learning rate.
              min_lr (float): minimal value of the learning rate
                  (same as ``end_learning_rate`` TensorFlow parameter).

            Returns:
              learning rate at step ``global_step``.
            """
            begin_decay_at = max(warmup_steps, begin_decay_at)
            if warmup_steps > 0:

                learning_rate = tf.cond(
                    global_step < warmup_steps,
                    lambda: (learning_rate * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)),
                    lambda: learning_rate,
                )
            lr = tf.cond(
                global_step < begin_decay_at,
                lambda: learning_rate,
                lambda: tf.train.polynomial_decay(
                    learning_rate,
                    global_step=global_step - begin_decay_at,
                    decay_steps=decay_steps,
                    end_learning_rate=min_lr,
                    power=power),
                name="learning_rate"
            )
            return lr

        lr_params = self.Config.lr_policy_params
        lr_params['decay_steps'] = self.last_step
        lr_policy = lambda gs: poly_decay(global_step=gs, **lr_params)
        return lr_policy

    def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
        """TensorFlow graph for Speech2Text model is created here.
        This function connects encoder, decoder and loss together. As an input for
        encoder it will specify source tensors (as returned from
        the data layer). As an input for decoder it will specify target tensors
        as well as all output returned from encoder. For loss it
        will also specify target tensors and all output returned from
        decoder. Note that loss will only be built for mode == "train" or "eval".

        Args:
            input_tensors (dict): ``input_tensors`` dictionary that has to contain
                ``source_tensors`` key with the list of all source tensors, and
                ``target_tensors`` with the list of all target tensors. Note that
                ``target_tensors`` only need to be provided if mode is "train" or "eval".
            gpu_id (int, optional): id of the GPU where the current copy of the model
                is constructed. For Horovod this is always zero.

        Returns:
            tuple: tuple containing loss tensor as returned from
            ``loss.compute_loss()`` and list of outputs tensors, which is taken from
            ``decoder.decode()['outputs']``. When ``mode == 'infer'``, loss will be None.
        """

        source_tensors = input_tensors['source_tensors']
        if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
            target_tensors = input_tensors['target_tensors']

        with tf.variable_scope('ForwardPass'):
            encoder_input = {'source_tensors': source_tensors}
            encoder_output = self._encoder.encode(input_dict=encoder_input)

            decoder_input = {'encoder_output': encoder_output}
            if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
                decoder_input['target_tensors'] = target_tensors
            decoder_output = self.decoder.decode(input_dict=decoder_input)
            model_outputs = decoder_output.get('outputs', None)

            if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
                with tf.variable_scope('Loss'):
                    loss_input_dict = {
                        'decoder_output': decoder_output,
                        'target_tensors': target_tensors,
                    }
                    loss = self.loss_computator.compute_loss(loss_input_dict)
            else:
                logger.log("Inference Mode. Loss part of graph isn't built.")
                loss = None
                if self.dump_outputs:
                    model_logits = decoder_output.get("logits", None)
                    return loss, [model_logits]
        return loss, model_outputs

    def _get_num_objects_per_step(self, worker_id=0):
        """Returns number of audio frames in current batch."""
        data_layer = self.get_data_layer(worker_id)
        num_frames = tf.reduce_sum(data_layer.input_tensors['source_tensors'][1])
        return num_frames

    def get_data_layer(self, worker_id=0):
        """Returns model data layer.
        Args:
          worker_id (int): id of the worker to get data layer from
        Returns:
          model data layer.
        """
        return self._data_layers[worker_id]

    def optimize_loss(self,
                      loss,
                      optimizer,
                      optimizer_params,
                      learning_rate_decay_fn,
                      var_list=None,
                      dtype=tf.float32,
                      clip_gradients=None,
                      summaries=None,
                      larc_params=None,
                      loss_scaling=1.0,
                      loss_scaling_params=None,
                      on_horovod=False,
                      iter_size=1,
                      skip_update_ph=None,
                      model=None):
        """Given loss and parameters for optimizer, returns a training op.

        Args:
          loss: Scalar `Tensor`.
          optimizer: string or class of optimizer, used as trainer.
              string should be name of optimizer, like 'SGD',
              'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
              class should be sub-class of `tf.Optimizer` that implements
              `compute_gradients` and `apply_gradients` functions.
          optimizer_params: parameters of the optimizer.
          var_list: List of trainable variables. Can be used to freeze
              certain trainable variables by excluding them from this list.
              If set to None, all trainable variables will be optimized.
          dtype: model dtype (tf.float16, tf.float32 or "mixed").
          learning_rate_decay_fn: function, takes `global_step`
              `Tensor`s, returns `Tensor`.
              Can be used to implement any learning rate decay
              functions.
              For example: `tf.train.exponential_decay`.
              Ignored if `learning_rate` is not supplied.
          clip_gradients: float, max gradient norm to clip to.
          summaries: List of internal quantities to visualize on tensorboard. If not
              set only the loss and the learning rate will be reported. The
              complete list is in OPTIMIZER_SUMMARIES.
          larc_params: If not None, LARC re-scaling will
              be applied with corresponding parameters.
          loss_scaling: could be float or string. If float, static loss scaling
              is applied. If string, the corresponding automatic
              loss scaling algorithm is used. Must be one of 'Backoff'
              of 'LogMax' (case insensitive). Only used when dtype="mixed".
          on_horovod: whether the model is run on horovod.

        Returns:
          training op.
        """
        OPTIMIZER_SUMMARIES = [
            "learning_rate",
            "gradients",
            "gradient_norm",
            "global_gradient_norm",
            "variables",
            "variable_norm",
            "larc_summaries",
            "loss_scale"
        ]

        if summaries is None:
            summaries = ["learning_rate", "global_gradient_norm", "loss_scale"]
        else:
            for summ in summaries:
                if summ not in OPTIMIZER_SUMMARIES:
                    raise ValueError(
                        "Summaries should be one of [{}], you provided {}.".format(
                            ", ".join(OPTIMIZER_SUMMARIES), summ,
                        )
                    )
        if clip_gradients is not None and larc_params is not None:
            raise AttributeError(
                "LARC and gradient norm clipping should not be used together"
            )

        global_step = tf.train.get_or_create_global_step()
        lr = learning_rate_decay_fn(global_step)
        if "learning_rate" in summaries:
            tf.summary.scalar("learning_rate", lr)

        with tf.variable_scope("Loss_Optimization"):
            update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            loss = control_flow_ops.with_dependencies(list(update_ops), loss)

            opt = optimizer(learning_rate=lr, **optimizer_params)

            if isinstance(loss_scaling, six.string_types) and loss_scaling == 'Backoff':
                loss_scaling = BackoffScaler(params=loss_scaling_params)
                if "loss_scale" in summaries:
                    tf.summary.scalar("loss_scale", loss_scaling.loss_scale)

            # Compute gradients.
            grads_and_vars = opt.compute_gradients(
                loss, colocate_gradients_with_ops=True, var_list=var_list
            )

            grad_updates = opt.apply_gradients(
                self.post_process_gradients(
                    grads_and_vars,
                    lr=lr,
                    clip_gradients=clip_gradients,
                    larc_params=larc_params,
                    summaries=summaries,
                ),
                global_step=global_step,
            )

            # Ensure the train_tensor computes grad_updates.
            train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)

            return train_tensor

    @staticmethod
    def post_process_gradients(grads_and_vars, summaries, lr,
                               clip_gradients, larc_params):
        """Applies post processing to gradients, i.e. clipping, LARC, summaries."""

        def _global_norm_with_cast(grads_and_vars):
            return tf.global_norm(list(map(
                lambda x: tf.cast(x, tf.float32),
                list(zip(*grads_and_vars))[0]
            )))

        def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
            """Clips gradients by global norm."""
            gradients, variables = zip(*grads_and_vars)
            dtypes = [var.dtype for var in variables]

            # Clip gradients in float32
            clipped_gradients, _ = _clip_by_global_norm(
                gradients,
                clip_gradients,
                use_norm=_global_norm_with_cast(grads_and_vars)
            )

            # Convert gradients back to the proper dtype
            clipped_gradients = [
                tf.cast(grad, dtype)
                for grad, dtype in zip(clipped_gradients, dtypes)
            ]

            return list(zip(clipped_gradients, variables))

        def _clip_by_global_norm(t_list, clip_norm, use_norm, name=None):
            """Clips values of multiple tensors by the ratio of the sum of their norms.
            Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
            this operation returns a list of clipped tensors `list_clipped`
            and the global norm (`global_norm`) of all tensors in `t_list`. The global
            norm is expected to be pre-computed and passed as use_norm.
            To perform the clipping, the values `t_list[i]` are set to:
                t_list[i] * clip_norm / max(global_norm, clip_norm)
            where:
                global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
            If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
            otherwise they're all shrunk by the global ratio.
            Any of the entries of `t_list` that are of type `None` are ignored.
            This is the correct way to perform gradient clipping (for example, see
            [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
            ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).
            However, it is slower than `clip_by_norm()` because all the parameters must be
            ready before the clipping operation can be performed.

            Args:
              t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
              clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
              use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
                norm to use. If not provided, `global_norm()` is used to compute the norm.
              name: A name for the operation (optional).

            Returns:
              list_clipped: A list of `Tensors` of the same type as `list_t`.
              global_norm: A 0-D (scalar) `Tensor` representing the global norm.

            Raises:
              TypeError: If `t_list` is not a sequence.
            """
            if (not isinstance(t_list, collections.Sequence)
                    or isinstance(t_list, six.string_types)):
                raise TypeError("t_list should be a sequence")
            t_list = list(t_list)

            # Removed as use_norm should always be passed
            # if use_norm is None:
            #   use_norm = global_norm(t_list, name)

            with tf.name_scope(name, "clip_by_global_norm",
                               t_list + [clip_norm]) as name:
                # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
                scale = clip_norm * tf.minimum(
                    1.0 / use_norm,
                    tf.ones([1], dtype=use_norm.dtype) / clip_norm)

                values = [
                    tf.cast(
                        tf.convert_to_tensor(
                            t.values if isinstance(t, tf.IndexedSlices) else t,
                            name="t_%d" % i, dtype=tf.float32),
                        dtype=tf.float32
                    )
                    if t is not None else t
                    for i, t in enumerate(t_list)]

                values_clipped = []
                for i, v in enumerate(values):
                    if v is None:
                        values_clipped.append(None)
                    else:
                        with tf.colocate_with(v):
                            values_clipped.append(
                                tf.identity(v * scale, name="%s_%d" % (name, i)))

                list_clipped = [
                    tf.IndexedSlices(c_v, t.indices, t.dense_shape)
                    if isinstance(t, tf.IndexedSlices)
                    else c_v
                    for (c_v, t) in zip(values_clipped, t_list)]

            return list_clipped, use_norm

        def mask_nans(x):
            x_zeros = tf.zeros_like(x)
            x_mask = tf.is_finite(x)
            y = tf.where(x_mask, x, x_zeros)
            return y

        if "global_gradient_norm" in summaries:
            tf.summary.scalar(
                "global_gradient_norm",
                _global_norm_with_cast(grads_and_vars),
            )

        # Optionally clip gradients by global norm.
        if clip_gradients is not None:
            grads_and_vars = _clip_gradients_by_norm(grads_and_vars, clip_gradients)

        # Add histograms for variables, gradients and gradient norms.
        for gradient, variable in grads_and_vars:
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient

            if isinstance(variable, tf.IndexedSlices):
                var_values = variable.values
            else:
                var_values = variable

            if grad_values is not None:
                var_name = variable.name.replace(":", "_")
                if "gradients" in summaries:
                    # need to mask nans for automatic loss scaling
                    tf.summary.histogram("gradients/%s" % var_name, mask_nans(grad_values))
                if "gradient_norm" in summaries:
                    tf.summary.scalar("gradient_norm/%s" % var_name, tf.norm(grad_values))
                if "variables" in summaries:
                    tf.summary.histogram("variables/%s" % var_name, var_values)
                if "variable_norm" in summaries:
                    tf.summary.scalar("variable_norm/%s" % var_name, tf.norm(var_values))

        if clip_gradients is not None and "global_gradient_norm" in summaries:
            tf.summary.scalar(
                "global_clipped_gradient_norm",
                _global_norm_with_cast(grads_and_vars),
            )

        # LARC gradient re-scaling
        if larc_params is not None:
            larc_eta = larc_params['larc_eta']
            larc_mode = larc_params.get('larc_mode', 'clip')
            min_update = larc_params.get('min_update', 1e-7)
            eps = larc_params.get('epsilon', 1e-7)

            grads_and_vars_larc = [None] * len(grads_and_vars)
            for idx, (g, v) in enumerate(grads_and_vars):
                var_dtype = v.dtype
                v_norm = tf.norm(tensor=tf.cast(v, tf.float32), ord=2)
                g_norm = tf.norm(tensor=tf.cast(g, tf.float32), ord=2)

                if larc_mode == 'clip':
                    larc_grad_update = tf.maximum(
                        larc_eta * v_norm / (lr * (g_norm + eps)),
                        min_update,
                    )
                    if "larc_summaries" in summaries:
                        tf.summary.scalar('larc_clip_on/{}'.format(v.name),
                                          tf.cast(tf.less(larc_grad_update, 1.0), tf.int32))
                    larc_grad_update = tf.minimum(larc_grad_update, 1.0)
                else:
                    larc_grad_update = tf.maximum(
                        larc_eta * v_norm / (g_norm + eps),
                        min_update,
                    )
                larc_grad_update = tf.saturate_cast(larc_grad_update, var_dtype)
                grads_and_vars_larc[idx] = (larc_grad_update * g, v)

                # adding additional summary
                if "larc_summaries" in summaries:
                    tf.summary.scalar('larc_grad_update/{}'.format(v.name),
                                      larc_grad_update)
                    tf.summary.scalar("larc_final_lr/{}".format(v.name),
                                      tf.cast(lr, var_dtype) * larc_grad_update)
            grads_and_vars = grads_and_vars_larc
        return grads_and_vars

    @staticmethod
    def get_regularization_loss(scope=None, name="total_regularization_loss"):
        """Gets the total regularization loss.

        Args:
          scope: An optional scope name for filtering the losses to return.
          name: The name of the returned tensor.

        Returns:
          A scalar regularization loss.
        """
        losses = tf.losses.get_regularization_losses(scope)
        if losses:
            return tf.add_n(list(map(lambda x: tf.cast(x, tf.float32), losses)),
                            name=name)
        else:
            return tf.constant(0.0)

    @property
    def encoder(self):
        """Model encoder."""
        return self._encoder

    @property
    def decoder(self):
        """Model decoder."""
        return self._decoder

    @property
    def loss_computator(self):
        """Model loss computator."""
        return self._loss_computator

    @property
    def steps_in_epoch(self):
        """Number of steps in epoch.
        This parameter is only populated if ``num_epochs`` was specified in the
        config (otherwise it is None).
        It is used in training hooks to correctly print epoch number.
        """
        return self._steps_in_epoch

    @property
    def last_step(self):
        """Number of steps the training should be run for."""
        return self._last_step


class TDNNEncoder:
    """General time delay neural network (TDNN) encoder. Fully convolutional model"""

    def __init__(self, params, Config, model, name="w2l_encoder"):
        self._model = model
        self._name = name
        self.params = params
        self.Config = Config

    def encode(self, input_dict):
        if 'initializer' in self.Config.encoder_params:
            init_dict = self.Config.encoder_params['initializer_params']
            initializer = self.Config.encoder_params['initializer'](**init_dict)
        else:
            initializer = None

        with tf.variable_scope(self._name, initializer=initializer,
                               dtype=self.Config.dtype):
            return self._encode(self.cast_types(input_dict, self.Config.dtype))

    def cast_types(self, input_dict, dtype):
        cast_input_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.float16 or value.dtype == tf.float32:
                    if value.dtype.base_dtype != dtype.base_dtype:
                        cast_input_dict[key] = tf.cast(value, dtype)
                        continue
            if isinstance(value, dict):
                cast_input_dict[key] = self.cast_types(input_dict[key], dtype)
                continue
            if isinstance(value, list):
                cur_list = []
                for nest_value in value:
                    if isinstance(nest_value, tf.Tensor):
                        if nest_value.dtype == tf.float16 or nest_value.dtype == tf.float32:
                            if nest_value.dtype.base_dtype != dtype.base_dtype:
                                cur_list.append(tf.cast(nest_value, dtype))
                                continue
                    cur_list.append(nest_value)
                cast_input_dict[key] = cur_list
                continue
            cast_input_dict[key] = input_dict[key]
        return cast_input_dict

    def _encode(self, input_dict):
        """Creates TensorFlow graph for Wav2Letter like encoder.

        Args:
          input_dict (dict): input dictionary that has to contain
              the following fields::
                input_dict = {
                  "source_tensors": [
                    src_sequence (shape=[batch_size, sequence length, num features]),
                    src_length (shape=[batch_size])
                  ]
                }

        Returns:
          dict: dictionary with the following tensors::

            {
              'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
              'src_length': tensor, shape=[batch_size]
            }
        """

        source_sequence, src_length = input_dict['source_tensors']

        num_pad = tf.constant(0)

        if isinstance(self._model.get_data_layer(), Speech2TextDataLayer):
            pad_to = self.Config.pad_to

            if pad_to > 0:
                num_pad = tf.mod(pad_to - tf.mod(tf.reduce_max(src_length), pad_to), pad_to)
        else:
            print("WARNING: TDNNEncoder is currently meant to be used with the",
                  "Speech2Text data layer. Assuming that this data layer does not",
                  "do additional padding past padded_batch.")

        max_len = tf.reduce_max(src_length) + num_pad

        training = (self.params['mode'] == 'train')
        encoder_params = self.Config.encoder_params
        dropout_keep_prob = encoder_params['dropout_keep_prob'] if training else 1.0
        regularizer = None
        data_format = encoder_params['data_format']
        normalization = encoder_params['normalization']

        drop_block_prob = 0
        drop_block_index = -1

        normalization_params = {'bn_momentum': 0.90, 'bn_epsilon': 1e-3}

        if encoder_params.get('use_conv_mask', False):
            mask = tf.sequence_mask(
                lengths=src_length, maxlen=max_len,
                dtype=source_sequence.dtype
            )
            mask = tf.expand_dims(mask, 2)

        conv_block = self.conv_bn_actv
        conv_inputs = source_sequence
        if data_format == 'channels_last':
            conv_feats = conv_inputs  # B T F
        else:
            conv_feats = tf.transpose(conv_inputs, [0, 2, 1])  # B F T

        residual_aggregation = []

        # ----- Convolutional layers ---------------------------------------------
        convnet_layers = encoder_params['convnet_layers']

        for idx_convnet in range(len(convnet_layers)):
            layer_type = convnet_layers[idx_convnet]['type']
            layer_repeat = convnet_layers[idx_convnet]['repeat']
            ch_out = convnet_layers[idx_convnet]['num_channels']
            kernel_size = convnet_layers[idx_convnet]['kernel_size']
            strides = convnet_layers[idx_convnet]['stride']
            padding = convnet_layers[idx_convnet]['padding']
            dilation = convnet_layers[idx_convnet]['dilation']
            dropout_keep = convnet_layers[idx_convnet].get(
                'dropout_keep_prob', dropout_keep_prob) if training else 1.0
            residual = convnet_layers[idx_convnet].get('residual', False)
            residual_dense = convnet_layers[idx_convnet].get('residual_dense', False)

            # For the first layer in the block, apply a mask
            if encoder_params.get("use_conv_mask", False):
                conv_feats = conv_feats * mask

            if residual:
                layer_res = conv_feats
                if residual_dense:
                    residual_aggregation.append(layer_res)
                    layer_res = residual_aggregation

            for idx_layer in range(layer_repeat):

                if padding == "VALID":
                    src_length = (src_length - kernel_size[0]) // strides[0] + 1
                    max_len = (max_len - kernel_size[0]) // strides[0] + 1
                else:
                    src_length = (src_length + strides[0] - 1) // strides[0]
                    max_len = (max_len + strides[0] - 1) // strides[0]

                # For all layers other than first layer, apply mask
                if idx_layer > 0 and encoder_params.get("use_conv_mask", False):
                    conv_feats = conv_feats * mask

                # Since we have a stride 2 layer, we need to update mask for future operations
                if (encoder_params.get("use_conv_mask", False) and
                        (padding == "VALID" or strides[0] > 1)):
                    mask = tf.sequence_mask(
                        lengths=src_length,
                        maxlen=max_len,
                        dtype=conv_feats.dtype
                    )
                    mask = tf.expand_dims(mask, 2)

                if residual and idx_layer == layer_repeat - 1:
                    conv_feats = self.conv_bn_res_bn_actv(
                        layer_type=layer_type,
                        name="conv{}{}".format(
                            idx_convnet + 1, idx_layer + 1),
                        inputs=conv_feats,
                        res_inputs=layer_res,
                        filters=ch_out,
                        kernel_size=kernel_size,
                        activation_fn=encoder_params['activation_fn'],
                        strides=strides,
                        padding=padding,
                        dilation=dilation,
                        regularizer=regularizer,
                        training=training,
                        data_format=data_format,
                        drop_block_prob=drop_block_prob,
                        drop_block=(drop_block_index == idx_convnet),
                        **normalization_params
                    )
                else:
                    conv_feats = conv_block(
                        layer_type=layer_type,
                        name="conv{}{}".format(
                            idx_convnet + 1, idx_layer + 1),
                        inputs=conv_feats,
                        filters=ch_out,
                        kernel_size=kernel_size,
                        activation_fn=encoder_params['activation_fn'],
                        strides=strides,
                        padding=padding,
                        dilation=dilation,
                        regularizer=regularizer,
                        training=training,
                        data_format=data_format,
                        **normalization_params
                    )

                conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)

        outputs = conv_feats

        if data_format == 'channels_first':
            outputs = tf.transpose(outputs, [0, 2, 1])

        return {
            'outputs': outputs,
            'src_length': src_length,
        }

    def conv_bn_actv(self, layer_type, name, inputs, filters, kernel_size, activation_fn,
                     strides, padding, regularizer, training, data_format,
                     bn_momentum, bn_epsilon, dilation=1):
        """Helper function that applies convolution, batch norm and activation.
          Args:
            layer_type: the following types are supported
              'conv1d', 'conv2d'
        """

        layers_dict = {
            "conv1d": tf.layers.conv1d,
            "conv2d": tf.layers.conv2d,
        }

        layer = layers_dict[layer_type]

        conv = layer(
            name="{}".format(name),
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation,
            kernel_regularizer=regularizer,
            use_bias=False,
            data_format=data_format,
        )

        # trick to make batchnorm work for mixed precision training.
        # To-Do check if batchnorm works smoothly for >4 dimensional tensors
        squeeze = False
        if layer_type == "conv1d":
            axis = 1 if data_format == 'channels_last' else 2
            conv = tf.expand_dims(conv, axis=axis)  # NWC --> NHWC
            squeeze = True

        bn = tf.layers.batch_normalization(
            name="{}/bn".format(name),
            inputs=conv,
            gamma_regularizer=regularizer,
            training=training,
            axis=-1 if data_format == 'channels_last' else 1,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
        )

        if squeeze:
            bn = tf.squeeze(bn, axis=axis)

        output = bn
        if activation_fn is not None:
            output = activation_fn(output)
        return output

    def conv_bn_res_bn_actv(self, layer_type, name, inputs, res_inputs, filters,
                            kernel_size, activation_fn, strides, padding,
                            regularizer, training, data_format, bn_momentum,
                            bn_epsilon, dilation=1,
                            drop_block_prob=0.0, drop_block=False):

        layers_dict = {
            "conv1d": tf.layers.conv1d,
            "conv2d": tf.layers.conv2d,
        }

        layer = layers_dict[layer_type]

        if not isinstance(res_inputs, list):
            res_inputs = [res_inputs]
            # For backwards compatibiliaty with earlier models
            res_name = "{}/res"
            res_bn_name = "{}/res_bn"
        else:
            res_name = "{}/res_{}"
            res_bn_name = "{}/res_bn_{}"

        res_aggregation = 0
        for i, res in enumerate(res_inputs):
            res = layer(
                res,
                filters,
                1,
                name=res_name.format(name, i),
                use_bias=False,
            )
            squeeze = False
            if layer_type == "conv1d":
                axis = 1 if data_format == 'channels_last' else 2
                res = tf.expand_dims(res, axis=axis)  # NWC --> NHWC
                squeeze = True
            res = tf.layers.batch_normalization(
                name=res_bn_name.format(name, i),
                inputs=res,
                gamma_regularizer=regularizer,
                training=training,
                axis=-1 if data_format == 'channels_last' else 1,
                momentum=bn_momentum,
                epsilon=bn_epsilon,
            )
            if squeeze:
                res = tf.squeeze(res, axis=axis)

            res_aggregation += res

        conv = layer(
            name="{}".format(name),
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation,
            kernel_regularizer=regularizer,
            use_bias=False,
            data_format=data_format,
        )

        # trick to make batchnorm work for mixed precision training.
        # To-Do check if batchnorm works smoothly for >4 dimensional tensors
        squeeze = False
        if layer_type == "conv1d":
            axis = 1 if data_format == 'channels_last' else 2
            conv = tf.expand_dims(conv, axis=axis)  # NWC --> NHWC
            squeeze = True

        bn = tf.layers.batch_normalization(
            name="{}/bn".format(name),
            inputs=conv,
            gamma_regularizer=regularizer,
            training=training,
            axis=-1 if data_format == 'channels_last' else 1,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
        )

        if squeeze:
            bn = tf.squeeze(bn, axis=axis)

        output = bn + res_aggregation

        if drop_block_prob > 0:
            if training:
                output = tf.cond(
                    tf.random_uniform(shape=[]) < drop_block_prob,
                    lambda: res_aggregation,
                    lambda: bn + res_aggregation
                )
            elif drop_block:
                output = res_aggregation

        if activation_fn is not None:
            output = activation_fn(output)
        return output


class FullyConnectedCTCDecoder:
    """Fully connected time decoder that provides a CTC-based text
    generation (either with or without language model). If language model is not
    used, ``tf.nn.ctc_greedy_decoder`` will be used as text generation method.
    """

    def __init__(self, params, Config, model, name="fully_connected_ctc_decoder"):
        """Fully connected CTC decoder constructor.

        See parent class for arguments description.

        Config parameters:

        * **use_language_model** (bool) --- whether to use language model for
          output text generation. If False, other config parameters are not used.
        * **decoder_library_path** (string) --- path to the ctc decoder with
          language model library.
        * **lm_path** (string) --- path to the language model file.
        * **trie_path** (string) --- path to the prefix trie file.
        * **alphabet_config_path** (string) --- path to the alphabet file.
        * **beam_width** (int) --- beam width for beam search.
        * **alpha** (float) --- weight that is assigned to language model
          probabilities.
        * **beta** (float) --- weight that is assigned to the
          word count.
        * **trie_weight** (float) --- weight for prefix tree vocabulary
          based character level rescoring.
        """
        self._model = model
        self._name = name
        self.params = params
        self.Config = Config

        def decode_without_lm(logits, decoder_input, merge_repeated=True):
            if logits.dtype.base_dtype != tf.float32:
                logits = tf.cast(logits, tf.float32)
            decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
                logits, decoder_input['encoder_output']['src_length'],
                merge_repeated,
            )
            return decoded

        self.params['logits_to_outputs_func'] = decode_without_lm

    def decode(self, input_dict):
        if 'initializer' in self.Config.decoder_params:
            init_dict = self.Config.decoder_params.get('initializer_params', {})
            initializer = self.Config.decoder_params['initializer'](**init_dict)
        else:
            initializer = None

        with tf.variable_scope(self._name, initializer=initializer,
                               dtype=self.Config.dtype):
            return self._decode(self.cast_types(input_dict, self.Config.dtype))

    def cast_types(self, input_dict, dtype):
        cast_input_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.float16 or value.dtype == tf.float32:
                    if value.dtype.base_dtype != dtype.base_dtype:
                        cast_input_dict[key] = tf.cast(value, dtype)
                        continue
            if isinstance(value, dict):
                cast_input_dict[key] = self.cast_types(input_dict[key], dtype)
                continue
            if isinstance(value, list):
                cur_list = []
                for nest_value in value:
                    if isinstance(nest_value, tf.Tensor):
                        if nest_value.dtype == tf.float16 or nest_value.dtype == tf.float32:
                            if nest_value.dtype.base_dtype != dtype.base_dtype:
                                cur_list.append(tf.cast(nest_value, dtype))
                                continue
                    cur_list.append(nest_value)
                cast_input_dict[key] = cur_list
                continue
            cast_input_dict[key] = input_dict[key]
        return cast_input_dict

    def _decode(self, input_dict):
        """Creates TensorFlow graph for fully connected time decoder.

        Args:
          input_dict (dict): input dictionary that has to contain
              the following fields::
                input_dict = {
                  'encoder_output': {
                    "outputs": tensor with shape [batch_size, time length, hidden dim]
                    "src_length": tensor with shape [batch_size]
                  }
                }

        Returns:
          dict: dictionary with the following tensors::

            {
              'logits': logits with the shape=[time length, batch_size, tgt_vocab_size]
              'outputs': logits_to_outputs_func(logits, input_dict)
            }
        """
        inputs = input_dict['encoder_output']['outputs']

        batch_size, _, n_hidden = inputs.get_shape().as_list()
        # Reshape from [B, T, A] --> [B*T, A].
        # Output shape: [n_steps * batch_size, n_hidden]
        inputs = tf.reshape(inputs, [-1, n_hidden])

        # Activation is linear by default
        tgt_vocab_size = self.Config.tgt_vocab_size
        logits = tf.layers.dense(
            inputs=inputs,
            units=tgt_vocab_size,
            kernel_regularizer=None,
            name='fully_connected',
        )
        logits = tf.reshape(
            logits,
            [batch_size, -1, tgt_vocab_size],
            name='logits',
        )
        # Converting to time_major=True shape
        if not (self.params['mode'] == 'infer' and self.Config.decoder_params.get('infer_logits_to_pickle', False)):
            logits = tf.transpose(logits, [1, 0, 2])
        if 'logits_to_outputs_func' in self.params:
            outputs = self.params['logits_to_outputs_func'](logits, input_dict)

            return {
                'outputs': outputs,
                'logits': logits,
                'src_length': input_dict['encoder_output']['src_length'],
            }
        return {
            'logits': logits,
            'src_length': input_dict['encoder_output']['src_length'],
        }


class CTCLoss:

    def __init__(self, name="ctc_loss"):
        self._name = name
        self.dtype = tf.float32

    def compute_loss(self, input_dict):
        with tf.variable_scope(self._name, dtype=self.dtype):
            return self._compute_loss(self.cast_types(input_dict, self.dtype))

    def cast_types(self, input_dict, dtype):
        cast_input_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.float16 or value.dtype == tf.float32:
                    if value.dtype.base_dtype != dtype.base_dtype:
                        cast_input_dict[key] = tf.cast(value, dtype)
                        continue
            if isinstance(value, dict):
                cast_input_dict[key] = self.cast_types(input_dict[key], dtype)
                continue
            if isinstance(value, list):
                cur_list = []
                for nest_value in value:
                    if isinstance(nest_value, tf.Tensor):
                        if nest_value.dtype == tf.float16 or nest_value.dtype == tf.float32:
                            if nest_value.dtype.base_dtype != dtype.base_dtype:
                                cur_list.append(tf.cast(nest_value, dtype))
                                continue
                    cur_list.append(nest_value)
                cast_input_dict[key] = cur_list
                continue
            cast_input_dict[key] = input_dict[key]
        return cast_input_dict

    def _compute_loss(self, input_dict):
        """CTC loss graph construction.

        Expects the following inputs::

          input_dict = {

          }

        Args:
          input_dict (dict): input dictionary that has to contain
              the following fields::
                input_dict = {
                  "decoder_output": {
                    "logits": tensor, shape [batch_size, time length, tgt_vocab_size]
                    "src_length": tensor, shape [batch_size]
                  },
                  "target_tensors": [
                    tgt_sequence (shape=[batch_size, time length, num features]),
                    tgt_length (shape=[batch_size])
                  ]
                }

        Returns:
          averaged CTC loss.
        """
        logits = input_dict['decoder_output']['logits']
        tgt_sequence, tgt_length = input_dict['target_tensors']
        # this loss needs an access to src_length since they
        # might get changed in the encoder
        src_length = input_dict['decoder_output']['src_length']

        # Compute the CTC loss
        total_loss = tf.nn.ctc_loss(
            labels=self._dense_to_sparse(tgt_sequence, tgt_length),
            inputs=logits,
            sequence_length=src_length,
            ignore_longer_outputs_than_inputs=True,
        )

        # Calculate the average loss across the batch
        avg_loss = tf.reduce_mean(total_loss)
        return avg_loss

    @staticmethod
    def _dense_to_sparse(dense_tensor, sequence_length):
        indices = tf.where(tf.sequence_mask(sequence_length))
        values = tf.gather_nd(dense_tensor, indices)
        shape = tf.shape(dense_tensor, out_type=tf.int64)
        return tf.SparseTensor(indices, values, shape)


class NovoGrad(MomentumOptimizer):
  """
  Optimizer that implements SGD with layer-wise normalized gradients,
  when normalization is done by sqrt(ema(sqr(grads))), similar to Adam

    ```
    Second moment = ema of Layer-wise sqr of grads:
       v_t <-- beta2*v_{t-1} + (1-beta2)*(g_t)^2

    First moment has two mode:
    1. moment of grads normalized by u_t:
       m_t <- beta1*m_{t-1} + lr_t * [ g_t/sqrt(v_t+epsilon)]
    1. moment similar to Adam: ema of grads normalized by u_t:
       m_t <- beta1*m_{t-1} + lr_t * [(1-beta1)*(g_t/sqrt(v_t+epsilon))]

    if weight decay add wd term after grads are rescaled by 1/sqrt(v_t):
       m_t <- beta1*m_{t-1} + lr_t * [g_t/sqrt(v_t+epsilon) + wd*w_{t-1}]

    Weight update:
       w_t <- w_{t-1} - *m_t
    ```

  """

  def __init__(self,
               learning_rate=1.0,
               beta1=0.95,
               beta2=0.98,
               epsilon=1e-8,
               weight_decay=0.0,
               grad_averaging=False,
               use_locking=False,
               name='NovoGrad'):
    """Constructor:

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      beta1: A `Tensor` or a float, used in ema for momentum.Default = 0.95.
      beta2: A `Tensor` or a float, used in ema for grad norms.Default = 0.99.
      epsilon: a float.  Default = 1e-8.
      weight_decay: A `Tensor` or a float, Default = 0.0.
      grad_averaging: switch between Momentum and SAG, Default = False,
      use_locking: If `True` use locks for update operations.
      name: Optional, name prefix for the ops created when applying
        gradients.  Defaults to "NovoGrad".
      use_nesterov: If `True` use Nesterov Momentum.

    """
    super(NovoGrad, self).__init__(learning_rate, momentum=beta1,
                                   use_locking=use_locking, name=name,
                                   use_nesterov=False)
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._wd  = weight_decay
    self._grad_averaging  = grad_averaging
    self._grads_ema = None



  def apply_gradients(self, grads_and_vars, global_step=None, name=None):

    # init ema variables if required
    len_vars = len(grads_and_vars)
    if self._grads_ema is None:
      self._grads_ema = [None] * len_vars
      for i in range(len_vars):
        self._grads_ema[i] = tf.get_variable(name="nvgrad2_ema" + str(i),
                                     shape=[], dtype=tf.float32,
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=False)

    # compute ema for grads^2 for each layer
    for i, (grad, var) in enumerate(grads_and_vars):
      g_2 = tf.reduce_sum(tf.square(x=tf.cast(grad, tf.float32)))
      self._grads_ema[i] = tf.cond(tf.equal(self._grads_ema[i], 0.),
                  lambda: g_2,
                  lambda: self._grads_ema[i]*self._beta2 + g_2*(1.-self._beta2)
                  )

      grad *= 1.0 / tf.sqrt(self._grads_ema[i] + self._epsilon)
      # weight decay
      if (self._wd > 0.):
        grad += (self._wd * var)
      # Momentum --> SAG
      if self._grad_averaging:
        grad *= (1.-self._beta1)
      grads_and_vars[i] = (grad, var)

    # call Momentum to do update
    return super(NovoGrad, self).apply_gradients(
         grads_and_vars, global_step=global_step, name=name)


class BackoffScaler(object):

    def __init__(self, params=None):
        if params is None:
            params = {}

        self.scale_min = params.get('scale_min', 1.0)
        self.scale_max = params.get('scale_max', 2.**14)
        self.step_factor = params.get('step_factor', 2.0)
        self.step_window = params.get('step_window', 2000)

        self.iteration = tf.Variable(initial_value=0,
                                     trainable=False,
                                     dtype=tf.int64)
        self.last_overflow_iteration = tf.Variable(initial_value=-1,
                                                   trainable=False,
                                                   dtype=tf.int64)
        self.scale = tf.Variable(initial_value=self.scale_max,
                                 trainable=False)

    def update_op(self, has_nan, amax):
        def overflow_case():
          new_scale_val = tf.clip_by_value(self.scale / self.step_factor,
                                           self.scale_min, self.scale_max)
          scale_assign = tf.assign(self.scale, new_scale_val)
          overflow_iter_assign = tf.assign(self.last_overflow_iteration,
                                           self.iteration)
          with tf.control_dependencies([scale_assign, overflow_iter_assign]):
            return tf.identity(self.scale)

        def scale_case():
          since_overflow = self.iteration - self.last_overflow_iteration
          should_update = tf.equal(since_overflow % self.step_window, 0)
          def scale_update_fn():
            new_scale_val = tf.clip_by_value(self.scale * self.step_factor,
                                             self.scale_min, self.scale_max)
            return tf.assign(self.scale, new_scale_val)
          return tf.cond(should_update,
                         scale_update_fn,
                         lambda: self.scale)

        iter_update = tf.assign_add(self.iteration, 1)
        overflow = tf.logical_or(has_nan, tf.is_inf(amax))

        update_op = tf.cond(overflow,
                            overflow_case,
                            scale_case)
        with tf.control_dependencies([update_op]):
          return tf.identity(iter_update)

    @staticmethod
    def check_grads(grads_and_vars):
        has_nan_ops = []
        amax_ops = []

        for grad, _ in grads_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    x = grad.values
                else:
                    x = grad

                has_nan_ops.append(tf.reduce_any(tf.is_nan(x)))
                amax_ops.append(tf.reduce_max(tf.abs(x)))

        has_nan = tf.reduce_any(has_nan_ops)
        amax = tf.reduce_max(amax_ops)
        return has_nan, amax

    @property
    def loss_scale(self):
      return self.scale


class PrintLossAndTimeHook(tf.train.SessionRunHook):
  """Session hook that prints training samples and prediction from time to time
  """
  def __init__(self, every_steps, model, print_ppl=False):
    super(PrintLossAndTimeHook, self).__init__()
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps)
    self._every_steps = every_steps
    self._iter_count = 0
    self._global_step = None
    self._model = model
    self._fetches = [model.loss]
    self._last_time = time.time()
    self._print_ppl = print_ppl

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()

  def before_run(self, run_context):
    if self._timer.should_trigger_for_step(self._iter_count):
      return tf.train.SessionRunArgs([self._fetches, self._global_step])
    return tf.train.SessionRunArgs([[], self._global_step])

  def after_run(self, run_context, run_values):
    results, step = run_values.results
    self._iter_count = step

    if not results:
      return
    self._timer.update_last_triggered_step(self._iter_count - 1)

    if self._model.steps_in_epoch is None:
      logger.log("Global step {}:".format(step), end=" ")
    else:
      logger.log(
          "Epoch {}, global step {}:".format(
              step // self._model.steps_in_epoch, step),
          end=" ",
      )

    loss = results[0]
    if not self._model.on_horovod or self._model.hvd.rank() == 0:
      if self._print_ppl:
        logger.log("Train loss: {:.4f} | ppl = {:.4f} | bpc = {:.4f}"
                   .format(loss, math.exp(loss),
                           loss/math.log(2)),
                   start="", end=", ")
      else:
        logger.log(
          "Train loss: {:.4f} ".format(loss),
          offset=4)

    tm = (time.time() - self._last_time) / self._every_steps
    m, s = divmod(tm, 60)
    h, m = divmod(m, 60)

    logger.log(
        "time per step = {}:{:02}:{:.3f}".format(int(h), int(m), s),
        start="",
    )
    self._last_time = time.time()


class TfJasper(BaseModel):
    '''
    Implements a speech recognition neural network model developed by Baidu. It contains five hiddlen layers.
    Validation set not implemented
    '''
    @staticmethod
    def get_knob_config():
        return {
            'epochs': FixedKnob(1),
            # batch_size should be no larger than the number of samples in the dataset
            'batch_size': CategoricalKnob([1]),
            'learning_rate': FloatKnob(1e-5, 1e-2, is_exp=True),
            'n_hidden': CategoricalKnob([128, 256, 512, 1024, 2048]),
            # lm_alpha and lm_beta can be used for further hyperparameter tuning
            # the alpha hyperparameter of the CTC decoder. Language Model weight
            'lm_alpha': FloatKnob(0.75, 0.75),
            # the beta hyperparameter of the CTC decoder. Word insertion weight
            'lm_beta': FloatKnob(1.85, 1.85)
        }

    @staticmethod
    def create_flags():
        # Importer
        # ========

        f = tf.app.flags

        f.DEFINE_integer('feature_win_len', 32, 'feature extraction audio window length in milliseconds')
        f.DEFINE_integer('feature_win_step', 20, 'feature extraction window step length in milliseconds')
        f.DEFINE_integer('audio_sample_rate', 16000, 'sample rate value expected by model')

        # Global Constants
        # ================

        f.DEFINE_float('dropout_rate', 0.05, 'dropout rate for feedforward layers')
        f.DEFINE_float('dropout_rate2', -1.0, 'dropout rate for layer 2 - defaults to dropout_rate')
        f.DEFINE_float('dropout_rate3', -1.0, 'dropout rate for layer 3 - defaults to dropout_rate')
        f.DEFINE_float('dropout_rate4', 0.0, 'dropout rate for layer 4 - defaults to 0.0')
        f.DEFINE_float('dropout_rate5', 0.0, 'dropout rate for layer 5 - defaults to 0.0')
        f.DEFINE_float('dropout_rate6', -1.0, 'dropout rate for layer 6 - defaults to dropout_rate')

        f.DEFINE_float('relu_clip', 20.0, 'ReLU clipping value for non-recurrent layers')

        # Printing

        f.DEFINE_integer('print_loss_steps', 10, 'number of steps to print loss')
        f.DEFINE_integer('print_samples_steps', 2200, 'number of steps to print loss')

        # Adam optimizer(http://arxiv.org/abs/1412.6980) parameters

        f.DEFINE_float('beta1', 0.9, 'beta 1 parameter of Adam optimizer')
        f.DEFINE_float('beta2', 0.999, 'beta 2 parameter of Adam optimizer')
        f.DEFINE_float('epsilon', 1e-8, 'epsilon parameter of Adam optimizer')

        # Checkpointing

        f.DEFINE_string('checkpoint_dir', '',
                        'directory in which checkpoints are stored - defaults to directory "/tmp/deepspeech/checkpoints" within user\'s home')
        # f.DEFINE_integer('checkpoint_secs', 600, 'checkpoint saving interval in seconds')
        f.DEFINE_integer("save_checkpoint_steps", 1100, 'checkpoint saving interval in steps')
        f.DEFINE_integer('max_to_keep', 3, 'number of checkpoint files to keep - default value is 5')

        # Exporting

        f.DEFINE_boolean('use_seq_length', True,
                         'have sequence_length in the exported graph(will make tfcompile unhappy)')

        # Reporting

        f.DEFINE_integer('report_count', 10,
                         'number of phrases with lowest WER(best matching) to print out during a WER report')

        # Initialization

        f.DEFINE_integer('random_seed', 4568, 'default random seed that is used to initialize variables')

        # Decoder

        f.DEFINE_string('alphabet_config_path', 'examples/datasets/speech_recognition/alphabet.txt',
                        'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')

        f.DEFINE_string('lm_binary_path', 'data/lm.binary',
                        'path to the language model binary file created with KenLM')
        f.DEFINE_string('lm_trie_path', 'data/trie',
                        'path to the language model trie file created with native_client/generate_trie')
        f.DEFINE_integer('beam_width', 1024,
                         'beam width used in the CTC decoder when building candidate transcriptions')
        f.DEFINE_float('lm_alpha', 0.75, 'the alpha hyperparameter of the CTC decoder. Language Model weight.')
        f.DEFINE_float('lm_beta', 1.85, 'the beta hyperparameter of the CTC decoder. Word insertion weight.')

    def initialize_globals(self):

        def get_available_gpus():
            r"""
            Returns the number of GPUs available on this system.
            """
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos if x.device_type == 'GPU']

        def load_pre_existing_vocabulary(path, min_idx=0, read_chars=False):
            """Loads pre-existing vocabulary into memory.
            Args:
              path (str): path to vocabulary.
              min_idx (int, optional): minimum id to assign for a token.
              read_chars (bool, optional): whether to read only the
                  first symbol of the line.

            Returns:
               dict: vocabulary dictionary mapping tokens (chars/words) to int ids.
            """
            idx = min_idx
            vocab_dict = {}
            with io.open(path, newline='', encoding='utf-8') as f:
                for line in f:
                    # ignoring empty line or commented lines
                    if not line or line == '\n' or line.startswith('#'):
                        continue
                    if line[0:2] == '\\#':
                        token = '#'
                    elif read_chars:
                        token = line[0]
                    else:
                        token = line.rstrip().split('\t')[0]
                    vocab_dict[token] = idx
                    idx += 1
            return vocab_dict

        c = AttrDict()

        # CPU device
        c.cpu_device = '/cpu:0'

        # Available GPU devices
        c.available_devices = get_available_gpus()

        # If there is no GPU available, we fall back to CPU based operation
        if not c.available_devices:
            c.available_devices = [c.cpu_device]

        c.char2idx = load_pre_existing_vocabulary(FLAGS.alphabet_config_path, read_chars=True)
        c.tgt_vocab_size = len(c.char2idx) + 1

        c.precompute_mel_basis = True
        c.input_type = 'logfbank'
        c.num_audio_features = 64

        c.batch_size = self._knobs.get('batch_size')

        c.dtype = tf.float16
        c.augmentation = {'n_freq_mask': 2, 'n_time_mask': 2, 'width_freq_mask': 6, 'width_time_mask': 6}
        c.train_max_duration = 16.7
        c.pad_to = 16

        c.encoder_params = {
            "convnet_layers": [
                {
                    "type": "conv1d", "repeat": 1,
                    "kernel_size": [11], "stride": [2],
                    "num_channels": 256, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.8,
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [11], "stride": [1],
                    "num_channels": 256, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.8,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [11], "stride": [1],
                    "num_channels": 256, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.8,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [13], "stride": [1],
                    "num_channels": 384, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.8,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [13], "stride": [1],
                    "num_channels": 384, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.8,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [17], "stride": [1],
                    "num_channels": 512, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.8,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [17], "stride": [1],
                    "num_channels": 512, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.8,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [21], "stride": [1],
                    "num_channels": 640, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.7,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [21], "stride": [1],
                    "num_channels": 640, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.7,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [25], "stride": [1],
                    "num_channels": 768, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.7,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 5,
                    "kernel_size": [25], "stride": [1],
                    "num_channels": 768, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.7,
                    "residual": True, "residual_dense": True
                },
                {
                    "type": "conv1d", "repeat": 1,
                    "kernel_size": [29], "stride": [1],
                    "num_channels": 896, "padding": "SAME",
                    "dilation":[2], "dropout_keep_prob": 0.6,
                },
                {
                    "type": "conv1d", "repeat": 1,
                    "kernel_size": [1], "stride": [1],
                    "num_channels": 1024, "padding": "SAME",
                    "dilation":[1], "dropout_keep_prob": 0.6,
                }
            ],

            "dropout_keep_prob": 0.7,

            "initializer": tf.contrib.layers.xavier_initializer,
            "initializer_params": {
                'uniform': False,
            },
            "normalization": "batch_norm",
            "activation_fn": tf.nn.relu,
            "data_format": "channels_last",
            "use_conv_mask": True,
        }

        c.decoder_params = {
            "initializer": tf.contrib.layers.xavier_initializer,
            "use_language_model": False,
            "infer_logits_to_pickle": False,
        }

        c.loss_params = {}

        c.loss_scaling = "Backoff"

        c.lr_policy_params = {
            "learning_rate": self._knobs.get('learning_rate'),
            "min_lr": 1e-5,
            "power": 2.0,
        }

        c.optimizer_params = {
            "beta1": 0.95,
            "beta2": 0.98,
            "epsilon": 1e-08,
            "weight_decay": 0.001,
            "grad_averaging": False,
        }

        c.larc_params = { "larc_eta": 0.001, }

        c.summaries = ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                       'variable_norm', 'gradient_norm', 'global_gradient_norm']

        c.save_summaries_steps = 100


        # Set default dropout rates
        if FLAGS.dropout_rate2 < 0:
            FLAGS.dropout_rate2 = FLAGS.dropout_rate
        if FLAGS.dropout_rate3 < 0:
            FLAGS.dropout_rate3 = FLAGS.dropout_rate
        if FLAGS.dropout_rate6 < 0:
            FLAGS.dropout_rate6 = FLAGS.dropout_rate

        # Set default checkpoint dir
        if not FLAGS.checkpoint_dir:
            FLAGS.checkpoint_dir = '/tmp/jasper/checkpoints'

        if not os.path.isdir(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)

        c.alphabet = Alphabet(FLAGS.alphabet_config_path)

        c.n_input = 26

        # The number of frames in theinitialize_globals context
        c.n_context = 9

        # Number of units in hidden layers
        c.n_hidden = self._knobs.get('n_hidden')

        c.n_hidden_1 = c.n_hidden

        c.n_hidden_2 = c.n_hidden

        c.n_hidden_5 = c.n_hidden

        # Units in the sixth layer = number of characters in the target language plus one
        c.n_hidden_6 = c.alphabet.size() + 1  # +1 for CTC blank label

        # LSTM cell state dimension
        c.n_cell_dim = c.n_hidden

        # The number of units in the third layer, which feeds in to the LSTM
        c.n_hidden_3 = c.n_cell_dim

        # Size of audio window in samples
        c.audio_window_samples = FLAGS.audio_sample_rate * (FLAGS.feature_win_len / 1000)

        # Stride for feature computations in samples
        c.audio_step_samples = FLAGS.audio_sample_rate * (FLAGS.feature_win_step / 1000)

        ConfigSingleton._config = c  # pylint: disable=protected-access

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        # self._sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
        #                                    inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
        # self._sess_config.gpu_options.allow_growth = True

        self.create_flags()
        self.f = tf.app.flags.FLAGS
        self.c = ConfigSingleton()
        self.initialize_globals()

    def _create_model(self, Config, params):
        return Speech2Text(Config, params)

    def train(self, dataset_uri):
        Config = self.c
        tf.set_random_seed(FLAGS.random_seed)
        with tf.Graph().as_default():
            params = {}
            params['dataset_uri'] = dataset_uri
            params['mode'] = 'train'
            dataset_dir = tempfile.TemporaryDirectory()
            params['dataset_dir'] = dataset_dir

            # Create train model
            train_model = self._create_model(self.c, params)
            train_model.compile()

            # Preparing to start training from scratch
            master_worker = True

            # Initializing session parameters
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            # pylint: disable=no-member
            sess_config.gpu_options.allow_growth = True

            # Defining necessary hooks
            hooks = [tf.train.StopAtStepHook(last_step=train_model.last_step)]

            if master_worker:
                checkpoint_dir = FLAGS.checkpoint_dir
                saver = tf.train.Saver(
                    save_relative_paths=False,
                    max_to_keep=FLAGS.max_to_keep
                )
                hooks.append(tf.train.CheckpointSaverHook(
                    checkpoint_dir,
                    saver=saver,
                    save_steps=FLAGS.save_checkpoint_steps
                ))

                if FLAGS.print_loss_steps:
                    hooks.append(PrintLossAndTimeHook(
                        every_steps=FLAGS.print_loss_steps,
                        model=train_model,
                    ))

            total_time = 0.0
            bench_start = 10
            init_data_layer = tf.group(
                [train_model.get_data_layer(i).iterator.initializer
                 for i in range(len(Config.available_devices))]
            )
            scaffold = tf.train.Scaffold(
                local_init_op=tf.group(tf.local_variables_initializer(), init_data_layer)
            )
            fetches = [train_model.train_op]

            total_objects = 0.0
            for worker_id in range(len(Config.available_devices)):
                fetches.append(train_model._get_num_objects_per_step(worker_id))

            # Start training
            sess = tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=checkpoint_dir,
                save_summaries_steps=Config.save_summaries_steps,
                config=sess_config,
                save_checkpoint_secs=None,
                log_step_count_steps=Config.save_summaries_steps,
                stop_grace_period_secs=300,
                hooks=hooks
            )
            step = 0
            num_bench_updates = 0
            while True:
                if sess.should_stop():
                    break
                tm = time.time()
                try:
                    feed_dict = {}
                    iter_size = 1
                    if step % iter_size == 0:
                        if step >= bench_start:
                            num_bench_updates += 1
                        fetches_vals = sess.run(fetches, feed_dict)
                    else:
                        # necessary to skip "no-update" steps when iter_size > 1
                        def run_with_no_hooks(step_context):
                            return step_context.session.run(fetches, feed_dict)
                        fetches_vals = sess.run_step_fn(run_with_no_hooks)
                except tf.errors.OutOfRangeError:
                    break
                if step >= bench_start:
                    total_time += time.time() - tm
                    if len(fetches) > 1:
                        for i in range(len(Config.available_devices)):
                            total_objects += np.sum(fetches_vals[i + 1])

                step += 1
            sess.close()

            if master_worker:
                logger.log('Finished training')
                if step > bench_start:
                    avg_time = 1.0 * total_time / num_bench_updates
                    logger.log('Avg time per step: {:.3f}s'.format(avg_time))
                    if len(fetches) > 1:
                        avg_objects = 1.0 * total_objects / total_time
                        logger.log("Avg objects per second: {:.3f}".format(avg_objects))


    def evaluate(self, dataset_uri):
        return 1

    def predict(self, queries, n_steps=16):
        pass

    def destroy(self):
        pass

    def dump_parameters(self):
        pass

    def load_parameters(self, params):
        pass


if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='TfJasper',
        task=TaskType.SPEECH_RECOGNITION,
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0',
        },
        # Demonstrative only, this dataset only contains one sample, we use batch_size = 1 to run
        # Replace with larger test data and larger batch_size in practice
        train_dataset_uri='data/ldc93s1/ldc93s1.zip',
        test_dataset_uri='data/ldc93s1/ldc93s1.zip',
        # Ensure the wav files have a sample rate of 16kHz
        queries=['data/ldc93s1/ldc93s1/LDC93S1.wav']
    )