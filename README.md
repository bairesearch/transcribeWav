# transcribeWav

### Author

Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

### Description

transcribes wav file to text using transformers Wav2Vec2Processor model 

### License

MIT License

### Installation
```
sudo apt-get install portaudio19-dev python3-pyaudio
conda create -n audiototext python=3.9
source activate audiototext
pip install transformers==4.11.2 soundfile sentencepiece torchaudio pyaudio
pip install pydub [required for audio split]
```

### Execution
```
ensure input wav file is 16000Hz and single channel (mono), e.g.;
	ffmpeg -i 'testFile1.mp4' -q:a 0 -map a 'testFile1.mp3'
	ffmpeg -i 'testFile1.mp3' -ar 16000 -ac 1 'testFile1.wav'
source activate audiototext
python3 transcribeWav.py
```
