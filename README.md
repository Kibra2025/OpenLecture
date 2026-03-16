# OpenLecture

OpenLecture is an open-source AI tool that transcribes lectures using Whisper.

## Features

- Speech-to-text transcription
- Multiple languages
- CLI interface
- AI-ready architecture

## Installation

pip install -r requirements.txt

## Usage

openlecture [audio.mp3]

By default, OpenLecture splits the input audio into 60-second chunks before
transcribing it. You can change the chunk size with:

openlecture .\lecture.mp3 --chunk-length-ms 120000
