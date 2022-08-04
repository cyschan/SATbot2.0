export function synthesiseSpeech(text, persona) {
    var sdk = require("microsoft-cognitiveservices-speech-sdk");
    const speechConfig = sdk.SpeechConfig.fromSubscription("112e652f93e840e9b130d38878a28a50", "uksouth");
    // Set either the `SpeechSynthesisVoiceName` or `SpeechSynthesisLanguage`.
    speechConfig.speechSynthesisLanguage = "en-GB";
    
    switch (String(persona)){
        case "Olivia":
            speechConfig.speechSynthesisVoiceName = "en-GB-SoniaNeural";
            break;
        case "Robert":
            speechConfig.speechSynthesisVoiceName = "en-GB-EthanNeural";
            break;
        case "Gabrielle":
            speechConfig.speechSynthesisVoiceName = "en-GB-HollieNeural";
            break;
        case "Arman":
            speechConfig.speechSynthesisVoiceName = "en-GB-ThomasNeural";
            break;
        default:
            return;
    }
    
    if (Array.isArray(text)){
        text = text.join('');
    }

    var synthesizer = new sdk.SpeechSynthesizer(speechConfig);
    synthesizer.speakTextAsync(
        text,
        result => {
            synthesizer.close();
            //return result.audioData;
            var context = new AudioContext();
            var source = context.createBufferSource();
            source.buffer = context.decodeAudioData(result.audioData);
            source.connect(context.destination);
            source.start();
        },
        error => {
            console.log(error);
            synthesizer.close();
        });

}