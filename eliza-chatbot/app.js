import { ElizaEngine } from "./engine.js";

const conversation = document.querySelector("#conversation");
const form = document.querySelector("#composer");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const welcome = document.querySelector("#welcome");
const prompts = document.querySelector("#prompts");
const newSession = document.querySelector("#new-session");
let engine = null;

function addMessage(text, role) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  conversation.append(bubble);
  bubble.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

async function startSession() {
  send.disabled = true;
  input.disabled = true;
  try {
    const response = await fetch("./doctor.script", { cache: "no-store" });
    if (!response.ok) throw new Error(`Script could not be loaded (HTTP ${response.status}).`);
    engine = new ElizaEngine(await response.text());
    conversation.replaceChildren();
    welcome.hidden = false;
    input.disabled = false;
    send.disabled = false;
    input.placeholder = "Write what's on your mind...";
    addMessage(engine.data.initial || "How do you do. Please tell me your problem.", "bot");
    input.focus();
  } catch (error) {
    engine = null;
    conversation.replaceChildren();
    welcome.hidden = true;
    addMessage(`Unable to start DOCTOR: ${error.message}`, "bot");
  }
}

function submitMessage(value) {
  const message = value.trim();
  if (!message || !engine || engine.closed) return;
  welcome.hidden = true;
  addMessage(message, "user");
  input.value = "";
  const result = engine.respond(message);
  addMessage(result.text, "bot");
  if (result.done) {
    input.disabled = true;
    send.disabled = true;
    input.placeholder = "Session ended. Start a new session.";
  }
  input.focus();
}

form.addEventListener("submit", event => {
  event.preventDefault();
  submitMessage(input.value);
});

prompts.addEventListener("click", event => {
  const button = event.target.closest("button[data-prompt]");
  if (button) submitMessage(button.dataset.prompt);
});

newSession.addEventListener("click", startSession);

startSession();