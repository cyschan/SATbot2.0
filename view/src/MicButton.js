import React, { useState } from "react";
import * as SpeechRecognition from "./SpeechRecognition";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMicrophone } from '@fortawesome/free-solid-svg-icons'

const MicButton = () => {
    const [isRecognitionStarted,setIsRecognitionStarted] = useState(false);

    const setSpeechRecognition = (e) => {
        console.log('isRecognitionStarted',isRecognitionStarted);
        if(!isRecognitionStarted)
        SpeechRecognition.startRecognition(e);
        else 
        SpeechRecognition.stopRecognition(e);

        setIsRecognitionStarted(!isRecognitionStarted);
    }
    return <a href="javascript:void(0)" onClick={(e) => setSpeechRecognition(e)}> 
    <FontAwesomeIcon icon={faMicrophone} style={{color:isRecognitionStarted ? '#60A2EA' : '#000000'}} /></a>
}

export default MicButton;