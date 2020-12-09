import speech_recognition as sr
import pyttsx3  


engine = pyttsx3.init()
#engine.setProperty('voice', voices[1].id)
listener = sr.Recognizer()


import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Audio file as source
# listening the audio file and store in audio_text variable

with sr.Microphone() as source:
    
    audio_text = r.listen(source)
    
    
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        
        # using google speech recognition
        voice = listener.listen(source)
        text = listener.recognize_google(voice).lower()
        print('Converting audio transcripts into text ...')

        print(text)

    except:
         print('Sorry.. run again...')