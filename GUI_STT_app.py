import io
import os
import sys
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from threading import Thread
from pydub import AudioSegment
import PyQt5.QtWidgets as qtw
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG
import speech_recognition as sr
from transformers import pipeline


class MyGUI(QWidget):
    def __init__(self):
        super().__init__()
        global pipe
        self.setWindowTitle('GUI Example')
        self.setGeometry(300, 300, 500, 300)  # Adjust the width and height values as desired

        # Create widgets
        self.label = QLabel('Voice To Text', self)
        self.label.setStyleSheet("font-size: 28px; font-weight: bold; color: #002244;")
        self.label.setAlignment(Qt.AlignCenter)

        self.record_button = QPushButton('Start Record', self)
        self.record_button.setStyleSheet(
            "background-color: #0066cc; color: #ffffff; padding: 12px; font-size: 20px; border-radius: 5px;")
        self.record_button.clicked.connect(self.toggle_transcription)

        self.pause_button = QPushButton('Pause Record', self)
        self.pause_button.setStyleSheet(
            "background-color: 'lightblue'; color: #ffffff; padding: 12px; font-size: 20px; border-radius: 5px;")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.button_save = QPushButton('Submit', self)
        self.button_save.setStyleSheet(
            "background-color: #808080; color: #ffffff; padding: 12px; font-size: 20px; border-radius: 5px;")
        self.button_save.clicked.connect(self.save_textbox)

        self.text_box = QTextEdit(self,
                                  lineWrapMode=QTextEdit.FixedColumnWidth,
                                  lineWrapColumnOrWidth=50,
                                  readOnly=False
                                  )
        self.text_box.setStyleSheet(
            "font-size: 18px; color: #ffffff; background-color: #002244; border: none; border-radius: 5px;")

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_box)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.record_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.button_save)

        layout.addLayout(buttons_layout)
        layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(layout)
        pipe = pipeline(model="BenShermaister/whisper-base-he", max_new_tokens=488)

    ## GLOBAL VARIABLES ##
    is_transcribing = False
    transcription = []
    text_box = None
    temp_file = None
    audio_segments = []  # Define audio_segments as an instance variable
    transcribe_thread = None
    is_paused = False

    def toggle_transcription(self):
        if self.is_transcribing:
            print("Stop Recording!")
            self.stop_transcription()
        else:
            print("Recording...")
            self.start_transcription()

    def start_transcription(self):
        self.is_transcribing = True
        self.audio_segments = []  # Clear audio_segments when starting a new transcription
        self.record_button.setText('Stop Record')
        self.record_button.setStyleSheet(
            "background-color: #ff0000; color: #ffffff; padding: 12px; font-size: 20px; border-radius: 5px;")
        self.pause_button.setEnabled(True)
        self.transcribe_thread = Thread(target=self.transcribe_audio)
        self.transcribe_thread.start()

    def stop_transcription(self):
        self.is_transcribing = False
        self.pause_button.setEnabled(False)
        self.record_button.setText('Start Record')
        self.record_button.setStyleSheet(
            "background-color: #0066cc; color: #ffffff; padding: 12px; font-size: 20px; border-radius: 5px;")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.setText("Resume")
            self.pause_button.setStyleSheet(
                "background-color: orange; color: #ffffff; padding: 12px; font-size: 20px; border-radius: 5px;")
        else:
            self.pause_button.setText("Pause")
            self.pause_button.setStyleSheet(
                "background-color: lightblue; color: #ffffff; padding: 12px; font-size: 20px; border-radius: 5px;")

    def save_textbox(self): #save txt and wav file iwth the same name to dataset folder
        text = self.text_box.toPlainText().replace("\n", " ")  # convert to string
        file_name = temp_file[-15:-4]  # files names
        file_name_text = file_name + ".txt"
        file_name_audio = file_name + ".wav"

        directory = "C:/pythonProject/pythonProjectHF/dataset"
        file_path_text = os.path.join(directory, file_name_text)
        file_path_wav = os.path.join(directory, file_name_audio)
        with open(file_path_text, 'w') as f:
            f.write(text)

        concatenated_audio = self.audio_segments[0]
        for segment in self.audio_segments[1:]:
            concatenated_audio += segment

        concatenated_audio.export(file_path_wav, format="wav")

    def transcribe_audio(self):
        global temp_file
        global is_paused

        data_queue = Queue()
        phrase_time = None
        last_sample = bytes()

        recorder = sr.Recognizer()
        recorder.energy_threshold = 800
        recorder.dynamic_energy_threshold = False

        source = sr.Microphone(sample_rate=32000)

        record_timeout = 2
        phrase_timeout = 3

        temp_file = NamedTemporaryFile().name + ".wav"
        transcription = ['']

        with source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            data_queue.put(data)

        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        while self.is_transcribing:
            try:
                global pipe
                now = datetime.utcnow()
                # pipe = pipeline(model="imvladikon/wav2vec2-xls-r-300m-hebrew", max_new_tokens=488)

                if not data_queue.empty():
                    phrase_complete = False

                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        last_sample = bytes()
                        phrase_complete = True

                    phrase_time = now

                    while not data_queue.empty():
                        data = data_queue.get()
                        last_sample += data

                    if self.is_paused:
                        continue

                    audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    with open(temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    self.audio_segments.append(AudioSegment.from_file(temp_file))

                    text = pipe(temp_file)["text"].strip()

                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    QMetaObject.invokeMethod(self.text_box, "clear", Qt.QueuedConnection)
                    for line in transcription:
                        QMetaObject.invokeMethod(self.text_box, "append", Qt.QueuedConnection, Q_ARG(str, line))
            except KeyboardInterrupt:
                break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MyGUI()
    gui.show()
    sys.exit(app.exec_())
