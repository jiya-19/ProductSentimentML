const MAX_LENGTH = 5000;

const textarea = document.getElementById("review-input");
const charCount = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const statusSection = document.getElementById("status-section");
const statusMessage = document.getElementById("status-message");
const resultSection = document.getElementById("result-section");
const resultSentiment = document.getElementById("result-sentiment");
const probBars = document.getElementById("prob-bars");

const CLASS_ORDER = ["Positive", "Neutral", "Negative"];

function updateCharCount() {
  const len = textarea.value.length;
  charCount.textContent = `${len} / ${MAX_LENGTH}`;
}

function showStatus(message, isError) {
  statusMessage.textContent = message;
  statusSection.hidden = false;
  statusSection.classList.toggle("error", Boolean(isError));
}

function hideStatus() {
  statusSection.hidden = true;
  statusSection.classList.remove("error");
}

function renderResult(data) {
  resultSentiment.textContent = data.sentiment;
  resultSentiment.className = `result-sentiment ${data.sentiment}`;

  probBars.innerHTML = "";
  CLASS_ORDER.forEach((cls) => {
    const value = data.probabilities[cls] ?? 0;
    const pct = Math.round(value * 100);

    const row = document.createElement("div");
    row.className = "prob-row";

    const name = document.createElement("span");
    name.className = "prob-name";
    name.textContent = cls;

    const track = document.createElement("div");
    track.className = "prob-track";
    const fill = document.createElement("div");
    fill.className = `prob-fill ${cls}`;
    fill.style.width = `${pct}%`;
    track.appendChild(fill);

    const pctLabel = document.createElement("span");
    pctLabel.className = "prob-pct";
    pctLabel.textContent = `${pct}%`;

    row.appendChild(name);
    row.appendChild(track);
    row.appendChild(pctLabel);
    probBars.appendChild(row);
  });

  resultSection.hidden = false;
}

async function analyzeSentiment() {
  const review = textarea.value.trim();

  if (!review) {
    showStatus("Enter a review before analyzing.", true);
    resultSection.hidden = true;
    return;
  }

  hideStatus();
  resultSection.hidden = true;
  analyzeBtn.disabled = true;
  showStatus("Analyzing review…", false);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review }),
    });

    if (!response.ok) {
      let detail = "Unable to analyze this review. Please try again.";
      try {
        const errBody = await response.json();
        if (errBody && errBody.detail) {
          detail = typeof errBody.detail === "string"
            ? errBody.detail
            : "Invalid input. Please check the review text.";
        }
      } catch (_) {
        /* response body wasn't JSON; keep default message */
      }
      showStatus(detail, true);
      return;
    }

    const data = await response.json();
    hideStatus();
    renderResult(data);
  } catch (err) {
    showStatus("Unable to connect to the prediction service. Please try again.", true);
  } finally {
    analyzeBtn.disabled = false;
  }
}

textarea.addEventListener("input", updateCharCount);
analyzeBtn.addEventListener("click", analyzeSentiment);
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    analyzeSentiment();
  }
});

updateCharCount();
