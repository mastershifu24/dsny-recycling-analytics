(function () {
  const startBtn = document.getElementById("startBtn");
  const sendBtn = document.getElementById("sendBtn");
  const textQ = document.getElementById("textQ");
  const transcriptP = document.getElementById("transcript");
  const resultP = document.getElementById("result");
  const speakBtn = document.getElementById("speakBtn");
  const stopSpeakBtn = document.getElementById("stopSpeakBtn");
  const autoReadEl = document.getElementById("autoRead");

  function maybeAutoRead(text) {
    if (!autoReadEl || !autoReadEl.checked || !window.speechSynthesis) return;
    const t = (text || "").trim();
    if (!t) return;
    if (/^(Thinking|Listening|Analyzing photo)/.test(t)) return;
    speakAnswer();
  }

  function setResult(text, placeholder) {
    resultP.textContent = text;
    resultP.classList.toggle("is-placeholder", !!placeholder);
    if (!placeholder) maybeAutoRead(text);
  }

  function speakAnswer() {
    if (!window.speechSynthesis) return;
    const txt = (resultP.textContent || "").trim();
    if (!txt || resultP.classList.contains("is-placeholder")) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(txt);
    utt.rate = 0.93;
    utt.lang = "en-US";
    const voices = window.speechSynthesis.getVoices();
    const best =
      voices.find(
        (v) => v.lang === "en-US" && (v.name.includes("Neural") || v.name.includes("Google"))
      ) || voices.find((v) => v.lang && v.lang.startsWith("en"));
    if (best) utt.voice = best;
    window.speechSynthesis.speak(utt);
  }

  if (speakBtn) speakBtn.addEventListener("click", speakAnswer);
  if (stopSpeakBtn)
    stopSpeakBtn.addEventListener("click", () => window.speechSynthesis && window.speechSynthesis.cancel());
  if (window.speechSynthesis) {
    window.speechSynthesis.addEventListener("voiceschanged", () => window.speechSynthesis.getVoices());
  } else if (speakBtn) {
    speakBtn.disabled = true;
    speakBtn.title = "Speech synthesis not available in this browser.";
  }

  setResult("Answers will appear here after you ask.", true);

  async function askBackend(question) {
    const q = (question || "").trim();
    if (!q) {
      setResult("Type a question above, or tap “Speak question.”", true);
      return;
    }
    transcriptP.textContent = "You asked: " + q;
    setResult("Thinking…", false);
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      setResult(data.answer || "No answer yet.", false);
    } catch (e) {
      setResult(
        "Request failed: " + (e && e.message ? e.message : String(e)),
        false
      );
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
  } else {
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;

    startBtn.addEventListener("click", () => {
      setResult("Listening…", false);
      recognition.start();
    });

    recognition.onerror = (ev) => {
      setResult("Speech error: " + (ev.error || "unknown"), false);
    };

    recognition.onresult = (event) => {
      const question = event.results[0][0].transcript;
      textQ.value = question;
      askBackend(question);
    };
  }

  const photoBtn = document.getElementById("photoBtn");
  const photoInput = document.getElementById("photoInput");
  const photoQ = document.getElementById("photoQ");
  if (photoBtn && photoInput) {
    photoBtn.addEventListener("click", async () => {
      const file = photoInput.files && photoInput.files[0];
      if (!file) {
        setResult("Choose a photo or video first.", true);
        return;
      }
      transcriptP.textContent = "";
      const isVid = file.type && file.type.startsWith("video/");
      setResult(isVid ? "Analyzing video (sampling frames)…" : "Analyzing photo…", false);
      const fd = new FormData();
      fd.append("image", file);
      if (photoQ && photoQ.value.trim()) {
        fd.append("question", photoQ.value.trim());
      }
      try {
        const res = await fetch("/analyze-image", { method: "POST", body: fd });
        const data = await res.json();
        if (data.error) {
          setResult(
            data.error + (data.disclaimer ? "\n\n" + data.disclaimer : ""),
            false
          );
          return;
        }
        let out = data.answer || "";
        if (data.media_kind === "video" && data.video_frames_used) {
          out =
            "[Video: used " +
            data.video_frames_used +
            " sampled frames]\n\n" +
            out;
        }
        if (data.disclaimer) {
          out += "\n\n" + data.disclaimer;
        }
        setResult(out || "No answer.", false);
      } catch (e) {
        setResult(
          "Request failed: " + (e && e.message ? e.message : String(e)),
          false
        );
      }
    });
  }
})();
