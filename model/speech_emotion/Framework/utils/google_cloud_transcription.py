import csv

from google.cloud import speech_v1p1beta1 as speech_v1
from google.cloud.speech_v1 import enums
from google.cloud import storage
import io
import os


def sample_long_running_recognize(storage_uri, transcript_path, multi_party=False):
    """
    Transcribe long audio file from Cloud Storage using asynchronous speech
    recognition

    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    client = speech_v1.SpeechClient()

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 32000

    # The language of the supplied audio
    language_code = "en-GB"

    if multi_party:
        # Enable time stamping
        enable_word_time_offsets = True
        nano_to_seconds = 1e-9
        enable_speaker_diarization = True
        diarization_speaker_count = 2

        # Encoding of audio data sent. This sample sets this explicitly.
        # This field is optional for FLAC and WAV audio formats.
        encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
        config = {
            "enable_word_time_offsets": enable_word_time_offsets,
            "sample_rate_hertz": sample_rate_hertz,
            "language_code": language_code,
            "encoding": encoding,
            "enable_speaker_diarization": enable_speaker_diarization,
            "diarization_speaker_count": diarization_speaker_count,
            # "audio_channel_count": 2,
        }
        audio = {"uri": storage_uri}

        operation = client.long_running_recognize(config, audio)

        print(u"Waiting for operation to complete...")
        response = operation.result()
        rows = []
        utterance = []
        curr_speaker = None
        utt_start = None
        utt_end = None
        for result in response.results:
            alternative = result.alternatives[0]
            confidence = alternative.confidence
            for word in alternative.words:
                if curr_speaker is None:
                    curr_speaker = word.speaker_tag
                if utt_start is None:
                    utt_start = word.start_time.seconds + (word.start_time.nanos * nano_to_seconds)
                if curr_speaker is not None and curr_speaker != word.speaker_tag:
                    # In this case the speaker has changed
                    row = (utt_start, utt_end, ' '.join(utterance), confidence, curr_speaker)
                    rows.append(row)
                    utt_start = word.start_time.seconds + (word.start_time.nanos * nano_to_seconds)
                    utterance = []

                utterance.append(word.word)
                utt_end = word.end_time.seconds + (word.end_time.nanos * nano_to_seconds)
                curr_speaker = word.speaker_tag
            if len(utterance) > 0:
                row = (utt_start, utt_end, ' '.join(utterance), confidence, curr_speaker)
                rows.append(row)
                utterance = []
        print(response)
        print(rows)
        with open(transcript_path, 'w', newline='') as csvfile:
            transcript_writer = csv.writer(csvfile, delimiter=',', quotechar='"')
            transcript_writer.writerow(['Start_Time', 'End_Time', 'Text', 'Confidence', 'Speaker'])
            for row in rows:
                start_time, end_time, words, confidence, speaker = row
                transcript_writer.writerow([start_time, end_time, words, confidence, speaker])
    else:
        # Enable time stamping
        enable_word_time_offsets = True
        nano_to_seconds = 1e-9

        # Encoding of audio data sent. This sample sets this explicitly.
        # This field is optional for FLAC and WAV audio formats.
        encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
        config = {
            "enable_word_time_offsets": enable_word_time_offsets,
            "sample_rate_hertz": sample_rate_hertz,
            "language_code": language_code,
            "encoding": encoding,
            # "audio_channel_count": 2,
        }
        audio = {"uri": storage_uri}

        operation = client.long_running_recognize(config, audio)

        print(u"Waiting for operation to complete...")
        response = operation.result()
        print(response)
        with open(transcript_path, 'w', newline='') as csvfile:
            transcript_writer = csv.writer(csvfile, delimiter=',', quotechar='"')
            for result in response.results:
                # First alternative is the most probable result
                alternative = result.alternatives[0]
                print(u"Transcript: {}".format(alternative.transcript))
                start_time = alternative.words[0].start_time.seconds + (alternative.words[0].start_time.nanos * nano_to_seconds)
                end_time = alternative.words[-1].end_time.seconds + (alternative.words[-1].end_time.nanos * nano_to_seconds)
                print(u"Time: ", start_time, end_time)
                speaker = alternative.words[0].speaker_tag
                transcript_writer.writerow([start_time, end_time, alternative.transcript, alternative.confidence, speaker])

def get_wav_files_and_save_paths(interview_dir):
    paths = []
    for dirpath, dirnames, filenames in os.walk(interview_dir):
        for filename in [f for f in filenames if f.endswith("audio_with_interviewer.wav")]:
            wav_file = os.path.join(dirpath, filename)
            transcript_save = os.path.join(dirpath, 'interviewer_transcript.csv')
            paths.append((wav_file, transcript_save, True))
        for filename in [f for f in filenames if f.endswith("audio.wav")]:
            wav_file = os.path.join(dirpath, filename)
            transcript_save = os.path.join(dirpath, 'transcript.csv')
            paths.append((wav_file, transcript_save, False))

    paths.sort(key=lambda pair: pair[0])
    return paths

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

# interview_dir = '/data/interviews'
# paths = get_wav_files_and_save_paths(interview_dir)
# for wav_file, transcript_save_path, multi_party in paths:
#     # Upload wav file to google cloud
#     uri_link = 'gs://er-audios/' + wav_file[17:]
#     reprocess = ['11', '24', '12']
#     skip = True
#     for id in reprocess:
#         if id in wav_file:
#             skip = False
#     if not multi_party or skip:
#         continue
#     print('Uploading {} to {}'.format(wav_file, uri_link))
#     upload_blob('er-audios', wav_file, wav_file[17:])
#     print('Transcribing (multi-party: {}) {} to {}'.format(multi_party, uri_link, transcript_save_path))
#     sample_long_running_recognize(uri_link, transcript_save_path, multi_party)

uri_link = 'gs://er-audios/25/audio_with_interviewer.wav'
sample_long_running_recognize(uri_link, 'test.csv', True)