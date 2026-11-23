"""
CPSC 254 - Assignment 4, Part 4 (2): RAG -> Text-to-speech (team version)

Sends a question through the team's Part 3 RAG application (rag.py, via ask_rag() in
stt_rag.py), converts the text answer to speech with the offline pyttsx3 engine (course example
NLP_TTS.py), saves it as a WAV file, and plays it aloud on the default speakers.

Usage
    python tts_rag.py "How is the final grade weighted between participation, assignments, and exams?"
    python tts_rag.py "..." --no-play --out audio/answer.wav   # no speakers available
"""

import argparse
import os
import re
import time
import wave

from stt_rag import ask_rag, print_rag_result


def speakable(text):
    """Remove markdown / symbols that sound bad when read aloud."""
    text = re.sub(r"[*_#`|>]+", " ", text)                # markdown symbols
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)   # bullet markers
    text = re.sub(r"\bpp?\.\s*(?=\d)", "page ", text)     # "p. 3" -> "page 3" (not "p.m.")
    text = text.replace("%", " percent")
    return " ".join(text.split())


def play_wav(path, engine=None, text=None):
    """Play the WAV on the default output device. (Re-using a pyttsx3 SAPI engine after
    save_to_file often produces no sound on Windows, so the saved file is played instead.)"""
    t0 = time.time()
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        dev = sd.query_devices(kind="output")
        print(f"Playing the answer aloud on '{dev['name']}'...")
        sd.play(data, sr)
        sd.wait()
    except Exception as exc:  # no audio device -> let pyttsx3 speak directly
        print(f"sounddevice playback unavailable ({exc}); using pyttsx3 say().")
        if engine is not None and text:
            engine.say(text)
            engine.runAndWait()
    print(f"Playback finished ({time.time() - t0:.1f}s).")


def speak(text, out_path, play=True, rate=165, voice_index=None):
    import pyttsx3

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    if voice_index is not None and voice_index < len(voices):
        engine.setProperty("voice", voices[voice_index].id)
    engine.setProperty("rate", rate)
    print(f"TTS engine: pyttsx3, voice: {engine.getProperty('voice')}, rate: {rate} wpm")
    engine.save_to_file(text, out_path)  # synthesize the answer to a WAV file
    engine.runAndWait()
    with wave.open(out_path, "rb") as w:
        dur = w.getnframes() / w.getframerate()
        rate_hz = w.getframerate()
    print(f"Saved spoken answer to {out_path} ({dur:.1f}s of audio, {rate_hz} Hz).")
    if play:
        play_wav(out_path, engine, text)


def main():
    ap = argparse.ArgumentParser(description="Syllabus RAG answer -> spoken audio")
    ap.add_argument("question", nargs="*", help="question for the RAG application")
    ap.add_argument("--out", default=os.path.join("audio", "answer.wav"))
    ap.add_argument("--no-play", action="store_true", help="only save the WAV file")
    ap.add_argument("--voice", type=int, default=None, help="pyttsx3 voice index (e.g. 1)")
    args = ap.parse_args()

    question = " ".join(args.question) or input("Question about the syllabus: ").strip()
    res = ask_rag(question)
    print_rag_result(question, res)
    print("=" * 78)
    spoken = speakable(res["answer"])
    print(f"TEXT SENT TO TTS: {spoken}")
    speak(spoken, args.out, play=not args.no_play, voice_index=args.voice)


if __name__ == "__main__":
    main()
