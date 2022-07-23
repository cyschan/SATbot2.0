import azure.cognitiveservices.speech as speechsdk
import pyaudio
import os
import wave

class SpeechSynthesiser:
    def __init__(self, persona):
        KEY = os.getenv('SYNTH_KEY')
        # Set either the `SpeechSynthesisVoiceName` or `SpeechSynthesisLanguage`.
        self.speech_config = speechsdk.SpeechConfig(subscription="112e652f93e840e9b130d38878a28a50", region="uksouth")
        self.speech_config.speech_synthesis_language = "en-US"
        self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        #self.audio_config = speechsdk.audio.AudioOutputConfig(filename="file.wav")
        self.persona = persona
        if persona == "Olivia":
            self.speech_config.speech_synthesis_voice_name = "en-GB-SoniaNeural"
        elif persona == "Arman":
            self.speech_config.speech_synthesis_voice_name = "en-GB-ThomasNeural"
        elif persona == "Robert":
            self.speech_config.speech_synthesis_voice_name = "en-GB-EthanNeural"
        elif persona == "Gabrielle":
            self.speech_config.speech_synthesis_voice_name = "en-GB-HollieNeural"
        else:
            self.speech_config.speech_synthesis_voice_name = "en-GB-SoniaNeural"

    def speak(self, text):
        if self.speech_config.speech_synthesis_voice_name == None:
            return
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=self.audio_config)
        synthesizer.speak_text_async(text)
        """stream = speechsdk.AudioDataStream(result)
        stream.save_to_wav_file_async("synth.wav")
        wf = wave.open('synth.wav', 'rb')
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
        # read data
        data = wf.readframes(CHUNK)

        # play stream (3)
        while len(data):
            stream.write(data)
            data = wf.readframes(CHUNK)

        # stop stream (4)
        stream.stop_stream()
        stream.close()

        # close PyAudio (5)
        p.terminate()"""

s = SpeechSynthesiser("Olivia")
s.speak("Hello World")
