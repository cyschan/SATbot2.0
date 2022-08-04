//const STT = require('stt');
//import Model from 'stt';
const Fs = require('fs');
const Sox = require('sox-stream');
const MemoryStream = require('memory-stream');
const Duplex = require('stream').Duplex;
const Wav = require('node-wav');

class SpeechRecogniser {
	constructor(modelPath = './src/speech_to_text/models/model.tflite', scorerPath = './src/speech_to_text/models/huge-vocabulary.scorer'){
		this.model = new Model(modelPath);
		this.desiredSampleRate = this.model.sampleRate();
		this.model.enableExternalScorer(scorerPath);
	}


	//let audioFile = process.argv[2] || './src/speech_to_text/audio/2830-3980-0043.wav';

	transcribe(audioFile){

		if (!Fs.existsSync(audioFile)) {
			console.log('file missing:', audioFile);
			process.exit();
		}

		const buffer = Fs.readFileSync(audioFile);
		const result = Wav.decode(buffer);

		if (result.sampleRate < this.desiredSampleRate) {
			console.error('Warning: original sample rate (' + result.sampleRate + ') is lower than ' + this.desiredSampleRate + 'Hz. Up-sampling might produce erratic speech recognition.');
		}

		function bufferToStream(buffer) {
			let stream = new Duplex();
			stream.push(buffer);
			stream.push(null);
			return stream;
		}

		let audioStream = new MemoryStream();
		bufferToStream(buffer).pipe(Sox({
			global: {
				'no-dither': true,
			},
			output: {
				bits: 16,
				rate: this.desiredSampleRate,
				channels: 1,
				encoding: 'signed-integer',
				endian: 'little',
				compression: 0.0,
				type: 'raw'
			}
		})).pipe(audioStream);

		audioStream.on('finish', () => {
			let audioBuffer = audioStream.toBuffer();		
			const audioLength = (audioBuffer.length / 2) * (1 / this.desiredSampleRate);
			console.log('audio length', audioLength);		
			let result = this.model.stt(audioBuffer);
			console.log('result:', result);
			return result;
		});
	}
}

let sr = new SpeechRecogniser();
sr.transcribe('./src/speech_to_text/audio/2830-3980-0043.wav');