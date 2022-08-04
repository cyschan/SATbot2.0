// MessageParser starter code

import * as SpeechRecognition from "./SpeechRecognition";

// const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
class MessageParser {
  constructor(actionProvider, state) {
    let mp = this;
    var audio = null,
      blob = null;
    this.mediaRecorder = null;
    let chunks = [];
    this.actionProvider = actionProvider;
    this.state = state;
    this.persona = null;
    this.recognition = SpeechRecognition?.returnRecognition();
    let speechData = undefined;
    this.recognition.onstart = function () {
      if (navigator.mediaDevices.getUserMedia) {
        //console.log('getUserMedia supported.');
        //let chunks = [];
        var options = {
          audioBitsPerSecond: 128000,
          mimeTyoe: "audio/webm",
        };
        navigator.mediaDevices
          .getUserMedia({ audio: true })
          .then(function (stream) {
            mp.mediaRecorder = new MediaRecorder(stream, options);
            mp.mediaRecorder.start();
          });
      } else {
        console.log("getUserMedia Unsupported.");
      }
    };

    this.recognition.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      speechData = e.results[0][0].transcript;
      mp.mediaRecorder.onstop = function (e) {
        console.log("data available after MediaRecorder.stop() called.", e);
        console.log("recorder stopped");
        var audioURL = null;
        audio = document.createElement("audio");
        console.log('audi0', audio);
        blob = new Blob(chunks, { type: "audio/webm; codecs=opus" });
        console.log('blob', blob);
        audioURL = window.URL.createObjectURL(blob);
        audio.src = audioURL;        
        actionProvider?.uploadSpeech(blob, speechData);
      };
      mp.mediaRecorder.ondataavailable = function (e) {
        console.log('ondata available', e.data);
        chunks.push(e.data);
      };
      mp.mediaRecorder.then = function () {
        if (blob != null) {
          mp.parse(transcript);
        }
      };
    };
    this.recognition.onend = (e) => {
      if(speechData !== undefined){
        mp.mediaRecorder.stop();
        this.recognition.abort();
        actionProvider.createClientMessage(speechData);
        actionProvider?.addSpokenMessage(speechData);
        // actionProvider?.uploadSpeech(audioURL, speechData);
        mp.parse(speechData, blob);
      }
    };
  }

  // This method is called inside the chatbot when it receives a message from the user.
  parse(message, audio = null) {
    // Case: User has not provided id yet
    if (this.state.username == null) {
      return this.actionProvider.askForPassword(message);
    } else if (this.state.password == null) {
      return this.actionProvider.updateUserID(this.state.username, message);
    } else if (
      this.state.askingForProtocol &&
      parseInt(message) >= 1 &&
      parseInt(message) <= 20
    ) {
      console.log(message);
      setTimeout(() => {
        this.recognition.start();
      }, 500);

      if (this.persona == null) {
        this.persona = this.actionProvider.getPersona(this.state.userState);
      }

      const choice_info = {
        user_id: this.state.userState,
        session_id: this.state.sessionID,
        user_choice: message,
        input_type: [
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ],
        persona: this.persona,
      };
      this.actionProvider.stopAskingForProtocol();
      return this.actionProvider.sendRequest(choice_info);
    } else if (
      this.state.askingForProtocol &&
      (parseInt(message) < 1 || parseInt(message) > 20)
    ) {
      console.log(message);
      setTimeout(() => {
        this.recognition.start();
      }, 500);

      return this.actionProvider.askForProtocol();
    } else {
      message = this.capitaliseFirstLetter(message);
      if (audio != null) {
        console.log(
          "inside parse",
          this.actionProvider.uploadSpeech(audio, message)
        );
        this.actionProvider.addSpokenMessage(message);
      }
      console.log(message);
      /*setTimeout(() => {
        this.recognition.start()
      }, 500)*/
      let input_type = null;
      if (this.state.inputType.length === 1) {
        if (
          this.state.messages[this.state.messages.length - 1].hasOwnProperty(
            "widget"
          ) &&
          this.state.messages[this.state.messages.length - 1].widget != null
        ) {
          input_type =
            this.state.messages[this.state.messages.length - 1].widget;
        } else {
          input_type = this.state.inputType[0];
        }
      } else {
        input_type = this.state.inputType;
      }
      const currentOptionToShow = this.state.currentOptionToShow;
      // Case: user types when they enter text instead of selecting an option
      if (
        (currentOptionToShow === "Continue" && message !== "Continue") ||
        (currentOptionToShow === "Emotion" &&
          message !== "Happy" &&
          message !== "Sad" &&
          message !== "Angry" &&
          message !== "Neutral") ||
        (currentOptionToShow === "RecentDistant" &&
          message !== "Recent" &&
          message !== "Distant") ||
        (currentOptionToShow === "Feedback" &&
          message !== "Better" &&
          message !== "Worse" &&
          message !== "No change") ||
        (currentOptionToShow === "Protocol" &&
          !this.state.protocols.includes(message)) ||
        (currentOptionToShow === "YesNo" &&
          message !== "Yes" &&
          message !== "No")
      ) {
        this.actionProvider.copyLastMessage();
      } else {
        const choice_info = {
          user_id: this.state.userState,
          session_id: this.state.sessionID,
          user_choice: message,
          input_type: input_type,
          persona: this.persona,
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
