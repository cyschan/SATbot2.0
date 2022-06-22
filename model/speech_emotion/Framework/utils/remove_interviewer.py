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
        print(response)
        print(rows)
        with open(transcript_path, 'w', newline='') as csvfile:
            transcript_writer = csv.writer(csvfile, delimiter=',', quotechar='"')
            transcript_writer.writerow(['Start_Time', 'End_Time', 'Text', 'Confidence', 'Speaker'])
            for row in rows:
                start_time, start_end, words, confidence, speaker = row
                transcript_writer.writerow([start_time, start_end, words, confidence, speaker])
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

def get_interviewer_transcripts(interview_dir):
    paths = []
    for dirpath, dirnames, filenames in os.walk(interview_dir):
        for filename in [f for f in filenames if f.endswith("interviewer_transcript.csv")]:
            csv_file = os.path.join(dirpath, filename)
            save_path = os.path.join(dirpath, filename[:-4] + '_new.csv')
            paths.append((csv_file, save_path))

    return paths

def remove_interviewer(csv_path, save_path):
    # Original stats.
    interviewer_id = 2 # this should be 1 or 2, usually 2
    count = 0
    with open(csv_path, "r") as fp:
        with open(save_path, 'w', newline='') as csvfile:
            transcript_writer = csv.writer(csvfile, delimiter=',', quotechar='"')
            transcript_writer.writerow(['Start_Time', 'End_Time', 'Text', 'Confidence', 'Speaker'])
            next(fp)  # skip the first line
            csv_reader = csv.reader(fp, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            for l in csv_reader:
                speaker_id = int(l[4].strip())

                start_time = float(l[0].strip())
                end_time = float(l[1].strip())
                text = l[2]
                confidence = float(l[3].strip())

                if 'read and understood' in text.lower() and speaker_id != 0:
                    # This is the interviewee
                    interviewer_id = speaker_id % 2 + 1
                    print('Interviewee ID detected as', speaker_id, 'changed interviewer id to', interviewer_id)

                if speaker_id == 0 or speaker_id == interviewer_id:
                    print('skipping', speaker_id)
                    continue

                count += 1
                transcript_writer.writerow([start_time, end_time, text, confidence, speaker_id])

    if count == 0:
        print('NO ROWS WRITTEN TO', save_path)


interview_dir = '/data/interviews'
paths = get_interviewer_transcripts(interview_dir)
for file_path, save_path in paths:
    print('removing interviewer from ', file_path, 'saving to ', save_path)
    remove_interviewer(file_path, save_path)

