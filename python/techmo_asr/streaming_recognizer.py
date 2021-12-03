import os
import threading
from ..techmo_service import dictation_asr_pb2 as dictation_asr_pb2
from ..techmo_service import dictation_asr_pb2_grpc as dictation_asr_pb2_grpc
import grpc
import logging

logger = logging.getLogger("techmo_asr")


class RequestIterator:
    """Thread-safe request iterator for streaming recognizer."""

    def __init__(self, o_streaming_recognizer, audio_generator):
        # Iterator data
        self.o_streaming_recognizer = o_streaming_recognizer
        self.audio_generator = audio_generator

        self.request_builder = {
            True: self._initial_request,
            False: self._normal_request
        }
        # Iterator state
        self.lock = threading.Lock()
        self.is_initial_request = True
        self.eos = False  # indicates whether end of stream message was send (request to stop iterator)

    def _initial_request(self):
        recognition_config = dictation_asr_pb2.RecognitionConfig(
            encoding=self.o_streaming_recognizer.encoding,  # one of LINEAR16, FLAC, MULAW, AMR, AMR_WB
            sample_rate_hertz=self.o_streaming_recognizer.sample_rate_hertz,  # the rate in hertz
            # See https://g.co/cloud/speech/docs/languages for a list of supported languages.
            language_code=self.o_streaming_recognizer.language_code,  # a BCP-47 language tag
            enable_word_time_offsets=False,  # if true, return recognized word time offsets
            max_alternatives=1,  # maximum number of returned hypotheses
        )
        if self.o_streaming_recognizer.speech_contexts:
            speech_context = recognition_config.speech_contexts.add()
            for context in self.o_streaming_recognizer.speech_contexts:
                speech_context.phrases.append(context)

        req = dictation_asr_pb2.StreamingRecognizeRequest(
            streaming_config=dictation_asr_pb2.StreamingRecognitionConfig(
                config=recognition_config,
                single_utterance=self.o_streaming_recognizer.single_utterance,
                interim_results=self.o_streaming_recognizer.interim_results,
            )
            # no audio data in first request (config only)
        )
        self.is_initial_request = False
        return req

    def _normal_request(self):
        data = next(self.audio_generator)
        if data is None:
            raise StopIteration

        return dictation_asr_pb2.StreamingRecognizeRequest(audio_content=data)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.request_builder[self.is_initial_request]()


class StreamingRecognizer:
    def __init__(self, address, ssl_directory, encoding, sample_rate_hertz, language_code, speech_contexts, interim_results=False, single_utterance=False, session_id=None):
        # Use ArgumentParser to parse settings
        self.service = dictation_asr_pb2_grpc.SpeechStub(StreamingRecognizer.create_channel(address, ssl_directory))
        self.encoding = encoding
        self.sample_rate_hertz = sample_rate_hertz
        self.language_code = language_code
        self.speech_contexts = speech_contexts
        self.interim_results = interim_results
        self.single_utterance = single_utterance
        self.session_id = session_id

    def recognize(self, audio_generator):
        requests_iterator = RequestIterator(self, audio_generator)
        return self.recognize_audio_content(requests_iterator)

    def recognize_audio_content(self, requests_iterator):
        metadata = []
        if self.session_id:
            metadata = [('session_id', self.session_id)]

        recognitions = self.service.StreamingRecognize(requests_iterator, metadata=metadata)

        transcript = ''
        confidence = None
        confidence_list = list()

        for recognition in recognitions:
            if recognition.error.code:
                logger.error(u"Received error response: ({}) {}".format(recognition.error.code, recognition.error.message))

            elif recognition.speech_event_type != dictation_asr_pb2.StreamingRecognizeResponse.SPEECH_EVENT_UNSPECIFIED:
                logger.error(u"Received speech event type: {}".format(dictation_asr_pb2.StreamingRecognizeResponse.SpeechEventType.Name(recognition.speech_event_type)))

            # process response type
            elif recognition.results is not None and len(recognition.results) > 0:
                first = recognition.results[0]
                if first.is_final:
                    transcript += first.alternatives[0].transcript
                    confidence_list.append(first.alternatives[0].confidence)

                else:
                    logger.debug(u"Temporal results - {}".format(first))

        if confidence_list:
            confidence = sum(confidence_list) / len(confidence_list)

        return transcript, confidence

    @staticmethod
    def create_channel(address, ssl_directory):
        if not ssl_directory:
            return grpc.insecure_channel(address)

        def read_file(path):
            with open(path, 'rb') as file:
                return file.read()

        return grpc.secure_channel(address, grpc.ssl_channel_credentials(
            read_file(os.path.join(ssl_directory, 'ca.crt')),
            read_file(os.path.join(ssl_directory, 'client.key')),
            read_file(os.path.join(ssl_directory, 'client.crt')),
        ))
