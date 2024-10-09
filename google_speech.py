import speech_recognition as sr
from googletrans import Translator

def recognize_and_translate(app):
    """
    Function to recognize speech from the microphone and translate the text to English.

    Args:
    app -- application object that is used to update the UI.

    Returns:
    A tuple containing the translated text in English and the original recognized text in Russian.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        app.print_text('Говорите...')
        app.update()
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        print('Listened, recognizing')
        app.print_text("Аудио распознается...")
        text = recognizer.recognize_google(audio, language='ru-RU')
        print('Recognized, translating...')
        if not text:
            return '', ''
        translator = Translator()
        translated_text_en = translator.translate(text, src='ru', dest='en').text
        return translated_text_en, text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return '', ''
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return '', ''