(function () {
  const startBtn = document.getElementById("startBtn");
  const sendBtn = document.getElementById("sendBtn");
  const textQ = document.getElementById("textQ");
  const transcriptP = document.getElementById("transcript");
  const resultP = document.getElementById("result");

  async function askBackend(question) {
    const q = (question || "").trim();
    if (!q) {
      resultP.textContent = "Enter a question or use the microphone.";
      return;
    }
    transcriptP.textContent = "You asked: " + q;
    resultP.textContent = "Thinking…";
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      resultP.textContent = data.answer || "No answer yet.";
    } catch (e) {
      resultP.textContent = "Request failed: " + (e && e.message ? e.message : String(e));
    }
  }

  sendBtn.addEventListener("click", () => askBackend(textQ.value));

  textQ.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      askBackend(textQ.value);
    }
  });

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    startBtn.disabled = true;
    startBtn.title = "Web Speech API not available in this browser.";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.continuous = false;
  recognition.interimResults = false;

  startBtn.addEventListener("click", () => {
    resultP.textContent = "Listening…";
    recognition.start();
  });

  recognition.onerror = (ev) => {
    resultP.textContent = "Speech error: " + (ev.error || "unknown");
  };

  recognition.onresult = (event) => {
    const question = event.results[0][0].transcript;
    textQ.value = question;
    askBackend(question);
  };

  const photoBtn = document.getElementById("photoBtn");
  const photoInput = document.getElementById("photoInput");
  const photoQ = document.getElementById("photoQ");
  if (photoBtn && photoInput) {
    photoBtn.addEventListener("click", async () => {
      const file = photoInput.files && photoInput.files[0];
      if (!file) {
        resultP.textContent = "Choose a photo first.";
        return;
      }
      transcriptP.textContent = "";
      resultP.textContent = "Analyzing photo…";
      const fd = new FormData();
      fd.append("image", file);
      if (photoQ && photoQ.value.trim()) {
        fd.append("question", photoQ.value.trim());
      }
      try {
        const res = await fetch("/analyze-image", { method: "POST", body: fd });
        const data = await res.json();
        if (data.error) {
          resultP.textContent =
            data.error + (data.disclaimer ? "\n\n" + data.disclaimer : "");
          return;
        }
        let out = data.answer || "";
        if (data.disclaimer) {
          out += "\n\n" + data.disclaimer;
        }
        resultP.textContent = out || "No answer.";
      } catch (e) {
        resultP.textContent =
          "Request failed: " + (e && e.message ? e.message : String(e));
      }
    });
  }
})();
