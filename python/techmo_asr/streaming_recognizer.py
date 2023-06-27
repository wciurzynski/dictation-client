import os
import threading
from techmo_service import dictation_asr_pb2 as dictation_asr_pb2
from techmo_service import dictation_asr_pb2_grpc as dictation_asr_pb2_grpc
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
        if self.o_streaming_recognizer.context_phrase:
            speech_context = recognition_config.speech_contexts.add()
            speech_context.phrases.append(self.o_streaming_recognizer.context_phrase)

        config_req = dictation_asr_pb2.StreamingRecognizeRequest(
            streaming_config=dictation_asr_pb2.StreamingRecognitionConfig(
                config=recognition_config,
                single_utterance=self.o_streaming_recognizer.single_utterance,
                interim_results=self.o_streaming_recognizer.interim_results,
            )
            # no audio data in first request (config only)
        )

        cf = config_req.streaming_config.config.config_fields.add()
        cf.key = 'no-input-timeout'
        cf.value = str(self.o_streaming_recognizer.no_input_timeout)

        cf = config_req.streaming_config.config.config_fields.add()
        cf.key = 'speech-complete-timeout'
        cf.value = str(self.o_streaming_recognizer.speech_complete_timeout)

        cf = config_req.streaming_config.config.config_fields.add()
        cf.key = 'speech-incomplete-timeout'
        cf.value = str(self.o_streaming_recognizer.speech_incomplete_timeout)

        cf = config_req.streaming_config.config.config_fields.add()
        cf.key = 'recognition-timeout'
        cf.value = str(self.o_streaming_recognizer.recognition_timeout)

        self.is_initial_request = False
        return config_req

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
    def __init__(
            self, address, encoding, sample_rate_hertz, language_code,
            context_phrase=None, interim_results=False, single_utterance=False, session_id=None,
            no_input_timeout=5000, speech_complete_timeout=2000, speech_incomplete_timeout=4000, recognition_timeout=10000,
            ssl_directory=None, root_certificates=None, private_key=None, certificate_chain=None, time_offsets=False
    ):
        # Use ArgumentParser to parse settings

        if ssl_directory:
            channel = StreamingRecognizer.create_channel(address, ssl_directory)
        elif root_certificates:
            channel = grpc.secure_channel(address, grpc.ssl_channel_credentials(root_certificates, private_key, certificate_chain))
        else:
            channel = grpc.insecure_channel(address)

        self.service = dictation_asr_pb2_grpc.SpeechStub(channel)
        self.encoding = encoding
        self.sample_rate_hertz = sample_rate_hertz
        self.language_code = language_code
        self.context_phrase = context_phrase
        self.interim_results = interim_results
        self.single_utterance = single_utterance
        self.session_id = session_id
        self.no_input_timeout = no_input_timeout
        self.speech_complete_timeout = speech_complete_timeout
        self.speech_incomplete_timeout = speech_incomplete_timeout
        self.recognition_timeout = recognition_timeout
        self.time_offsets = time_offsets

    def recognize(self, audio_generator):
        requests_iterator = RequestIterator(self, audio_generator)
        return self.recognize_audio_content(requests_iterator)

    def recognize_audio_content(self, requests_iterator):
        metadata = []
        if self.session_id:
            metadata = [('session_id', self.session_id)]

        recognitions = self.service.StreamingRecognize(requests_iterator, metadata=metadata)

        confirmed_results = []
        alignment = []
        confidence = None
        confidence_list = list()

        for recognition in recognitions:
            if recognition.error.code:
                logger.error(u"Received error response: ({}) {}".format(recognition.error.code, recognition.error.message))

            elif recognition.speech_event_type != dictation_asr_pb2.StreamingRecognizeResponse.SPEECH_EVENT_UNSPECIFIED:
                logger.error(u"Received speech event type: {}".format(dictation_asr_pb2.StreamingRecognizeResponse.SpeechEventType.Name(recognition.speech_event_type)))

            elif recognition.results is not None and len(recognition.results) > 0:
                first = recognition.results[0]
                if first.is_final:
                    if self.time_offsets:
                        for word in first.alternatives[0].words:
                            if word.word != '<eps>':
                                confirmed_results.append(word.word)
                                alignment.append([word.start_time, word.end_time])

                    else:
                        confirmed_results.append(first.alternatives[0].transcript)

                    confidence_list.append(first.alternatives[0].confidence)

                else:
                    logger.debug(u"Temporal results - {}".format(first))

        if confidence_list:
            confidence = sum(confidence_list) / len(confidence_list)

        final_alignment = [[]]
        final_transcript = ' '.join(confirmed_results)

        if self.time_offsets:
            if alignment:
                final_alignment = alignment

            return {
                'transcript': final_transcript,
                'alignment': final_alignment,
                'confidence': confidence
            }

        else:
            return final_transcript, confidence

    @staticmethod
    def create_channel(address, ssl_directory):
        if not ssl_directory:
            return grpc.insecure_channel(address)

        if not os.path.isdir(ssl_directory):
            logging.error(f'SSL directory {ssl_directory} not exists')
            return grpc.insecure_channel(address)

        def read_file(path):
            with open(path, 'rb') as file:
                return file.read()

        return grpc.secure_channel(address, grpc.ssl_channel_credentials(
            read_file(os.path.join(ssl_directory, 'ca.crt')),
            read_file(os.path.join(ssl_directory, 'client.key')),
            read_file(os.path.join(ssl_directory, 'client.crt')),
        ))
