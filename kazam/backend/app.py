import io

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from signal_matcher.signal import Signal
from signal_matcher.sound import Sound
from song_solver import SongSolver

solver = SongSolver("../songs")

app = FastAPI()

# CORS policy for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:9000",
        "http://127.0.0.1:9000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/match-song")
async def upload_file(audio: UploadFile = File(...)):
    contents = await audio.read()

    audio_buffer = io.BytesIO(contents)
    sound = Sound(audio_buffer=audio_buffer)
    result = solver.cross_correlation_solve(sound, Signal.pearson_correlation, 'auto')

    return {
        "match": result.best_match.reference.name,
        "confidence": float(result.best_match.confidence),
        "match-2": result.second_best_match.reference.name,
        "confidence-2": float(result.second_best_match.confidence),
    }
