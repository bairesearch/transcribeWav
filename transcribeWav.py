"""transcribeWav.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
sudo apt-get install portaudio19-dev python3-pyaudio
conda create -n audiototext python=3.9
source activate audiototext
pip install transformers==4.11.2 soundfile sentencepiece torchaudio pyaudio
pip install pydub [required for audio split]

# Usage:
ensure input wav file is 16000Hz and single channel (mono), e.g.;
	ffmpeg -i 'testFile1.mp4' -q:a 0 -map a 'testFile1.mp3'
	ffmpeg -i 'testFile1.mp3' -ar 16000 -ac 1 'testFile1.wav'
source activate audiototext
python3 transcribeWav.py

# Description:
transcribeWav - transcribes wav file to text using transformers Wav2Vec2Processor model

get_transcription() is derived from https://github.com/x4nth055/pythoncode-tutorials/blob/master/machine-learning/nlp/speech-recognition-transformers/AutomaticSpeechRecognition_PythonCodeTutorial.py

"""

from transformers import *
import torch
import soundfile as sf
import os
import torchaudio
import math
from pydub import AudioSegment

audio_url = "testFile1.wav"

# model_name = "facebook/wav2vec2-base-960h" # 360MB
model_name = "facebook/wav2vec2-large-960h-lv60-self" # 1.18GB

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

segmentTimeS = 60	#50	#10	#s	#depends on available RAM
firstSegmentIndex = 0	#default = 0

def transcribeAudio(audio_path):

	transcribedText = ""
	audioFile = AudioSegment.from_wav(audio_path)	
	#audioFile = AudioSegment.from_mp3(audio_path)
	audioFileTimeS = audioFile.duration_seconds
	numberOfSegments = math.ceil(audioFileTimeS/segmentTimeS)
	print("audioFileTimeS = ", audioFileTimeS, ", numberOfSegments = ", numberOfSegments)
	for timeIndex in range(firstSegmentIndex, numberOfSegments):
		t1S = timeIndex*segmentTimeS
		t2S = t1S+segmentTimeS
		t1Ms = t1S*1000
		t2Ms = t2S*1000
		print("timeIndex = ", timeIndex, ", t1S = ", t1S, ", t2S = ", t2S)
		audioSegment = audioFile[t1Ms:t2Ms]
		audioSegmentName = "audioSegment" + str(timeIndex) + ".wav"
		audioSegment.export(audioSegmentName, format="wav")
		transcribedSegmentText = get_transcription(audioSegmentName)
		print("transcribedSegmentText = ", transcribedSegmentText)
		transcribedText = transcribedText + " " + transcribedSegmentText
	print("transcribedText = ", transcribedText)

#derived from https://github.com/x4nth055/pythoncode-tutorials/blob/master/machine-learning/nlp/speech-recognition-transformers/AutomaticSpeechRecognition_PythonCodeTutorial.py	
def get_transcription(audio_path):
	speech, sr = torchaudio.load(audio_path)
	speech = speech.squeeze()
	resampler = torchaudio.transforms.Resample(sr, 16000)
	speech = resampler(speech)
	input_values = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
	logits = model(input_values)["logits"]
	predicted_ids = torch.argmax(logits, dim=-1)
	transcription = processor.decode(predicted_ids[0])
	return transcription.lower()

if __name__ == "__main__":
	transcribeAudio(audio_url)
