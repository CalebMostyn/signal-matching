function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

let mediaRecorder;
let audioChunks = [];

const startBtn = document.getElementById("record-button");
const textbox = document.getElementById("textbox");

startBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    textbox.textContent = "";

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, {
        type: "audio/webm",
      });

      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.webm");

      const response = await fetch("http://localhost:8000/match-song", {
        method: "POST",
        body: formData,
        credentials: "include"
      });

      data = await response.json();
      textbox.textContent = `Song: ${data.match} - Confidence: ${data.confidence.toFixed(2) * 100}%`;
    };

    mediaRecorder.start();
    startBtn.disabled = true;
    await sleep(5000)
    mediaRecorder.stop();
    startBtn.disabled = false;

  } catch (err) {
    console.error("Microphone access denied:", err);
  }
});
