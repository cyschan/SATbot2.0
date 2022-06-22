// MessageParser starter code
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
/*class MessageParser {
  constructor(actionProvider, state) {
    this.actionProvider = actionProvider;
    this.state = state;

    this.recognition = new SpeechRecognition()
    this.recognition.continous = false
    this.recognition.interimResults = false
    this.recognition.lang = 'en-US'
    this.recognition.maxAlternatives = 1;

    this.recognition.start()
    this.recognition.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      this.recognition.abort()
      this.parse(transcript)
    }
  }

  parse(message) {
    console.log(message)
    setTimeout(() => {
      this.recognition.start()
    }, 500)
    return this.actionProvider.askForPassword(message);
  }
}

export default MessageParser*/
class MessageParser {
  constructor(actionProvider, state) {
    let mp = this;
    var audio = null, blob = null, audioURL = null;
    this.mediaRecorder = null;
    let chunks = [];
    this.actionProvider = actionProvider;
    this.state = state;
    this.recognition = new SpeechRecognition()
    this.recognition.continuous = false
    this.recognition.interimResults = false
    this.recognition.lang = 'en-US'
    this.recognition.maxAlternatives = 1;
    this.recognition.start()
    this.recognition.onstart = function(){
      if (navigator.mediaDevices.getUserMedia) {
        //console.log('getUserMedia supported.');
        let chunks = [];
        var options = {
          audioBitsPerSecond: 128000,
          mimeTyoe: 'audio/webm'
        }
        navigator.mediaDevices.getUserMedia({ audio: true, })
          .then(function (stream) {
            mp.mediaRecorder = new MediaRecorder(stream, options);
            mp.mediaRecorder.start();
          })
      } else {
        console.log('getUserMedia Unsupported.');
      }
    };

    this.recognition.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      var audioURL = null;
      mp.mediaRecorder.stop();
      this.recognition.abort()
      mp.mediaRecorder.onstop = function (e) {
        console.log("data available after MediaRecorder.stop() called.");
        audio = document.createElement('audio');
        blob = new Blob(chunks, { 'type': 'audio/webm; codecs=opus' });
        audioURL = window.URL.createObjectURL(blob);
        audio.src = audioURL;
        console.log("recorder stopped");
        const recording = new Audio(audioURL)
        recording.play()
        mp.parse(transcript, blob)
      }
      mp.mediaRecorder.ondataavailable = function (e) {
        chunks.push(e.data);
      }
      mp.mediaRecorder.then = function(){
        if (blob != null){
          mp.parse(transcript)
        }
      }
    }
  }


  /*if (navigator.mediaDevices.getUserMedia) {
    console.log('getUserMedia supported.');
    let chunks = [];
    var options = {
      audioBitsPerSecond: 128000,
      mimeTyoe: 'audio/wav'
    }
    navigator.mediaDevices.getUserMedia( { audio: true, } )
    .then(function (stream) {
      mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorder.start();
      setTimeout(function () {
        mediaRecorder.stop()
    }, 5000);
      mediaRecorder.onstop = function (e) {
        console.log("data available after MediaRecorder.stop() called.");
        audio = document.createElement('audio');
        blob = new Blob(chunks, { 'type': 'audio/wav; codecs=0' });
        audioURL = window.URL.createObjectURL(blob);
        audio.src = audioURL;
        console.log("recorder stopped");
        const recording = new Audio(audioURL)
        recording.play()
      }
      mediaRecorder.ondataavailable = function(e) {
        chunks.push(e.data);
      }
      
    }
    )*/


  // This method is called inside the chatbot when it receives a message from the user.
  parse(message, audio = null) {
    // Case: User has not provided id yet

    if (this.state.username == null) {
      return this.actionProvider.askForPassword(message);
    } else if (this.state.password == null) {
      return this.actionProvider.updateUserID(this.state.username, message);
    } else if (this.state.askingForProtocol && parseInt(message) >= 1 && parseInt(message) <= 20) {
      console.log(message)
      setTimeout(() => {
        this.recognition.start()
      }, 500)
      const choice_info = {
        user_id: this.state.userState,
        session_id: this.state.sessionID,
        user_choice: message,
        input_type: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      };
      this.actionProvider.stopAskingForProtocol()

      return this.actionProvider.sendRequest(choice_info);
    } else if (this.state.askingForProtocol && (parseInt(message) < 1 || parseInt(message) > 20)) {
      console.log(message)
      setTimeout(() => {
        this.recognition.start()
      }, 500)

      return this.actionProvider.askForProtocol()
    }
    else {
      message = this.capitaliseFirstLetter(message)
      if (audio != null){
        console.log(this.actionProvider.uploadSpeech(audio, message))
        this.actionProvider.addSpokenMessage(message)
      }
      console.log(message)
      /*setTimeout(() => {
        this.recognition.start()
      }, 500)*/
      let input_type = null;
      if (this.state.inputType.length === 1) {
        if (this.state.messages[this.state.messages.length - 1].hasOwnProperty('widget') && this.state.messages[this.state.messages.length - 1].widget != null){
          input_type = this.state.messages[this.state.messages.length - 1].widget
        } else {
          input_type = this.state.inputType[0]
        }
      } else {
        input_type = this.state.inputType
      }
      const currentOptionToShow = this.state.currentOptionToShow
      // Case: user types when they enter text instead of selecting an option
      if ((currentOptionToShow === "Continue" && message !== "Continue") ||
        (currentOptionToShow === "Emotion" && (message !== "Happy" && message !== "Sad" && message !== "Angry" && message !== "Neutral")) ||
        (currentOptionToShow === "RecentDistant" && (message !== "Recent" && message !== "Distant")) ||
        (currentOptionToShow === "Feedback" && (message !== "Better" && message !== "Worse" && message !== "No change")) ||
        (currentOptionToShow === "Protocol" && (!this.state.protocols.includes(message))) ||
        (currentOptionToShow === "YesNo" && (message !== "Yes" && message !== "No"))
      ) {
        this.actionProvider.copyLastMessage()
      } else {
        const choice_info = {
          user_id: this.state.userState,
          session_id: this.state.sessionID,
          user_choice: message,
          input_type: input_type,
        };



        return this.actionProvider.sendRequest(choice_info);
      }
    }
  }

  capitaliseFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
  }

}

export default MessageParser;