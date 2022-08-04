const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.continuous = false;
recognition.interimResults = false;
recognition.lang = "en-US";
recognition.maxAlternatives = 1;

export const startRecognition = (e) => {
  if (e) {
    recognition.start();
  }
};

export const returnRecognition = () => {
  return recognition;
};

export const stopRecognition = (e) => {
  if (e) {
    recognition.stop();
  }
};
