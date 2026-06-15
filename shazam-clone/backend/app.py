import io

from fastapi import FastAPI, UploadFile, File

from signal_matcher.signal import Signal
from signal_matcher.sound import Sound
from song_solver import SongSolver

solver = SongSolver("../songs")

app = FastAPI()

@app.post("/match-song")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    audio_buffer = io.BytesIO(contents)
    sound = Sound(audio_buffer=audio_buffer)
    result = solver.cross_correlation_solve(sound, Signal.pearson_correlation, 'auto')

    return {
        "match": result.best_match.reference.name,
        "confidence": float(result.best_match.confidence),
        "match-2": result.second_best_match.reference.name,
        "confidence-2": float(result.second_best_match.confidence),
    }
