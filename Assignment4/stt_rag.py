"""
CPSC 254 - Assignment 4, Part 4 (1): Speech-to-text -> RAG (team version)

Captures a spoken question (microphone) or loads an audio file, transcribes it with OpenAI
Whisper (runs locally; based on the course example NLP_STT_whisper.py), sends the transcription
to the team's Part 3 RAG application (rag.py), and displays both the transcription and the answer.

rag.py (Member C) only has to provide ONE of these interfaces; ask_rag() below detects which:
    class SyllabusRAG with .ask(question)
    get_vector_store(), get_llm(), ask(question, store, llm)
    answer_question(question)  or  ask(question)
    or module-level vectorstore, prompt and llm objects (a script-style rag.py)
Each may return a string or a dict containing "answer".
Until rag.py exists, the placeholder rag_stub.py is used, and a warning is printed.

Usage
    python stt_rag.py --audio audio/question1.wav      # audio-file workflow
    python stt_rag.py --mic --seconds 6                 # record from the microphone
    python stt_rag.py --make-sample "What are the instructor's office hours?" --out audio/question1.wav
"""

import argparse
import os
import time

import numpy as np

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono float32


# ----------------------------------------------------------------------------
# Connection to the team's RAG application (shared with tts_rag.py)
# ----------------------------------------------------------------------------
def _load_rag_module():
    import contextlib
    import io

    try:
        # rag.py may run a demo question when imported; hide that printout
        with contextlib.redirect_stdout(io.StringIO()):
            import rag
        return rag, False
    except ModuleNotFoundError as exc:
        if exc.name != "rag":  # rag.py exists but one of ITS imports is missing: show the real error
            raise
        import rag_stub
        print("WARNING: rag.py not found - using the placeholder rag_stub.py. "
              "Re-run after Member C's rag.py is in this folder.")
        return rag_stub, True


def ask_rag(question):
    """Send a question to the team's RAG app; return {'answer', 'pages', 'seconds', 'stub'}."""
    rag, is_stub = _load_rag_module()
    t0 = time.time()
    if hasattr(rag, "SyllabusRAG"):
        res = rag.SyllabusRAG().ask(question)
    elif all(hasattr(rag, n) for n in ("get_vector_store", "get_llm", "ask")):
        res = rag.ask(question, rag.get_vector_store(), rag.get_llm())
    elif hasattr(rag, "answer_question"):
        res = rag.answer_question(question)
    elif hasattr(rag, "ask"):
        res = rag.ask(question)
    elif all(hasattr(rag, n) for n in ("vectorstore", "prompt", "llm")):
        # script-style rag.py: reuse its vector store, prompt and LLM for the new question
        docs = rag.vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        msg = rag.llm.invoke(rag.prompt.format_messages(context=context, question=question))
        res = {"answer": msg.content,
               "pages": sorted({int(d.metadata.get("page", 0)) + 1 for d in docs})}
    else:
        raise AttributeError("rag.py has no SyllabusRAG, ask(), answer_question() or "
                             "vectorstore/prompt/llm; ask Member C which function answers a question.")
    if isinstance(res, str):
        res = {"answer": res}
    answer = res.get("answer") or res.get("result") or str(res)
    pages = res.get("pages") or sorted({s.get("page") for s in res.get("sources", []) if s.get("page")})
    return {"answer": answer.strip(), "pages": pages, "seconds": round(time.time() - t0, 2),
            "stub": is_stub}


def print_rag_result(question, res):
    print(f"\nQUESTION: {question}")
    print(f"ANSWER:   {res['answer']}")
    extra = f"retrieved syllabus page(s) {res['pages']}; " if res["pages"] else ""
    print(f"({extra}RAG time {res['seconds']}s{'; PLACEHOLDER STUB' if res['stub'] else ''})")


# ----------------------------------------------------------------------------
# Audio input
# ----------------------------------------------------------------------------
def make_sample(text, out_path):
    """Synthesize a spoken question to a WAV file (for environments without a microphone)."""
    import pyttsx3

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    print(f"Wrote spoken question to {out_path}: \"{text}\"")


def load_audio_file(path):
    """Read a WAV/FLAC/OGG file into 16 kHz mono float32 (no ffmpeg needed)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    print(f"Loaded {path}: {len(audio) / SAMPLE_RATE:.1f}s of audio (original {sr} Hz)")
    return audio


def record_mic(seconds):
    import sounddevice as sd

    print(f"Recording for {seconds} seconds... ask your syllabus question now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.flatten()


def transcribe(audio, model_name="base"):
    import whisper

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak  # normalise volume like the course example
    t0 = time.time()
    model = whisper.load_model(model_name)
    text = model.transcribe(audio, fp16=False, language="en")["text"].strip()
    print(f"Whisper '{model_name}' transcription finished in {time.time() - t0:.1f}s")
    return text


def main():
    ap = argparse.ArgumentParser(description="Speech question -> Whisper STT -> syllabus RAG")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--audio", help="path to a WAV/FLAC/OGG file containing the question")
    src.add_argument("--mic", action="store_true", help="record the question from the microphone")
    src.add_argument("--make-sample", metavar="TEXT", help="create a spoken sample question and exit")
    ap.add_argument("--out", default=os.path.join("audio", "question1.wav"), help="output for --make-sample")
    ap.add_argument("--seconds", type=float, default=6, help="recording length for --mic")
    ap.add_argument("--whisper-model", default="base", help="tiny/base/small/... (default base)")
    args = ap.parse_args()

    if args.make_sample:
        make_sample(args.make_sample, args.out)
        return
    audio = record_mic(args.seconds) if args.mic else load_audio_file(args.audio or args.out)

    question = transcribe(audio, args.whisper_model)
    print("=" * 78)
    print(f"TRANSCRIPTION: {question}")
    print("=" * 78)
    if not question:
        print("No speech was recognised; nothing sent to the RAG application.")
        return
    print_rag_result(question, ask_rag(question))


if __name__ == "__main__":
    main()
