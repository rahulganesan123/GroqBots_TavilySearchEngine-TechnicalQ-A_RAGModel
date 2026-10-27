"""IndusDocs Web UI — Lightweight local web interface for testing the RAG assistant.

Uses Python's standard library http.server (no extra pip packages required).
Run:
    python ui.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from app import (
    DATA_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    TAVILY_API_KEY,
    LocalPdfRetriever,
    load_pdf_chunks,
    query_rag,
)

PORT = int(os.getenv("PORT", "8000"))
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IndusDocs — Technical RAG Assistant</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --sidebar-bg: #0b1120;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --accent: #6366f1;
      --bubble-user: #1e3a8a;
      --bubble-bot: #1e293b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    /* Sidebar */
    .sidebar {
      width: 320px;
      background: var(--sidebar-bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }
    .sidebar-header {
      padding: 20px;
      border-bottom: 1px solid var(--border);
    }
    .sidebar-header h1 {
      font-size: 1.15rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #fff;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-top: 8px;
      padding: 4px 10px;
      background: #064e3b;
      color: #34d399;
      font-size: 0.75rem;
      border-radius: 9999px;
      font-weight: 600;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      background: #34d399;
      border-radius: 50%;
      box-shadow: 0 0 6px #34d399;
    }
    .sidebar-section {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      overflow-y: auto;
    }
    .sidebar-section h2 {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 10px;
      font-weight: 700;
    }
    .file-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 180px;
      overflow-y: auto;
    }
    .file-item {
      font-size: 0.82rem;
      color: #cbd5e1;
      padding: 6px 10px;
      background: var(--card-bg);
      border-radius: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .file-icon { color: var(--primary); }
    .quick-prompts {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .prompt-btn {
      text-align: left;
      font-size: 0.8rem;
      padding: 8px 12px;
      background: var(--card-bg);
      color: #e2e8f0;
      border: 1px solid var(--border);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .prompt-btn:hover {
      border-color: var(--primary);
      background: #334155;
      color: #fff;
    }
    /* Main Chat */
    .main-chat {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100vh;
      position: relative;
    }
    .chat-header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--sidebar-bg);
    }
    .chat-header h2 {
      font-size: 1rem;
      font-weight: 600;
    }
    .clear-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: 0.2s;
    }
    .clear-btn:hover {
      background: #ef444422;
      border-color: #ef4444;
      color: #f87171;
    }
    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .message {
      display: flex;
      gap: 14px;
      max-width: 85%;
      animation: fadeIn 0.2s ease-in;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .message.user {
      align-self: flex-end;
      flex-direction: row-reverse;
    }
    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.85rem;
      flex-shrink: 0;
    }
    .avatar.user { background: var(--accent); color: #fff; }
    .avatar.bot { background: var(--primary); color: #042f2e; }
    .bubble {
      padding: 14px 18px;
      border-radius: 12px;
      font-size: 0.92rem;
      line-height: 1.6;
    }
    .message.user .bubble {
      background: var(--bubble-user);
      color: #fff;
      border-bottom-right-radius: 2px;
    }
    .message.bot .bubble {
      background: var(--bubble-bot);
      border: 1px solid var(--border);
      border-bottom-left-radius: 2px;
      color: #f1f5f9;
    }
    .message.bot .bubble p { margin-bottom: 10px; }
    .message.bot .bubble p:last-child { margin-bottom: 0; }
    .message.bot .bubble ul, .message.bot .bubble ol {
      margin-left: 20px;
      margin-bottom: 10px;
    }
    .message.bot .bubble li { margin-bottom: 4px; }
    .message.bot .bubble strong { color: var(--primary); }
    .source-tag {
      display: inline-block;
      margin-top: 8px;
      padding: 2px 8px;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 4px;
      font-size: 0.72rem;
      color: var(--text-muted);
    }
    /* Thinking Indicator */
    .typing-indicator {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 10px 14px;
    }
    .typing-indicator span {
      width: 8px;
      height: 8px;
      background: var(--primary);
      border-radius: 50%;
      animation: bounce 1.4s infinite ease-in-out both;
    }
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0); }
      40% { transform: scale(1); }
    }
    /* Input Form */
    .input-container {
      padding: 18px 24px;
      background: var(--sidebar-bg);
      border-top: 1px solid var(--border);
    }
    .input-form {
      display: flex;
      gap: 10px;
      position: relative;
    }
    .chat-input {
      flex: 1;
      padding: 14px 18px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: #fff;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .chat-input:focus { border-color: var(--primary); }
    .send-btn {
      padding: 0 24px;
      background: var(--primary);
      color: #042f2e;
      border: none;
      border-radius: 10px;
      font-weight: 700;
      font-size: 0.92rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    .send-btn:hover { background: var(--primary-hover); }
    .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
</head>
<body>
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <h1>🛠️ IndusDocs RAG</h1>
      <div class="status-badge">
        <div class="status-dot"></div>
        <span id="model-name">Groq Active</span>
      </div>
    </div>

    <div class="sidebar-section">
      <h2>Indexed Documents (<span id="doc-count">0</span>)</h2>
      <ul class="file-list" id="file-list">
        <li class="file-item">Loading manuals...</li>
      </ul>
    </div>

    <div class="sidebar-section" style="flex: 1;">
      <h2>Suggested Questions</h2>
      <div class="quick-prompts">
        <button class="prompt-btn" onclick="sendPrompt(this.innerText)">what is welding torches</button>
        <button class="prompt-btn" onclick="sendPrompt(this.innerText)">What is the rating and wire size for M250?</button>
        <button class="prompt-btn" onclick="sendPrompt(this.innerText)">What cooling method is used for M18?</button>
        <button class="prompt-btn" onclick="sendPrompt(this.innerText)">What are the features of the B-Series MIG torch?</button>
        <button class="prompt-btn" onclick="sendPrompt(this.innerText)">hi</button>
      </div>
    </div>
  </aside>

  <!-- Main Chat -->
  <main class="main-chat">
    <div class="chat-header">
      <h2>Technical Q&amp;A Workspace</h2>
      <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
    </div>

    <div class="chat-messages" id="chat-messages">
      <div class="message bot">
        <div class="avatar bot">ID</div>
        <div class="bubble">
          <p>Hello! I am <strong>IndusDocs</strong>, your technical document assistant.</p>
          <p>I can instantly answer questions and extract technical specifications from the PDFs stored in your <code>data/</code> folder.</p>
        </div>
      </div>
    </div>

    <div class="input-container">
      <form class="input-form" id="chat-form" onsubmit="handleSubmit(event)">
        <input
          type="text"
          id="user-input"
          class="chat-input"
          placeholder="Ask a question about your technical manuals... (Press Enter to send)"
          autocomplete="off"
        />
        <button type="submit" id="send-btn" class="send-btn">Ask</button>
      </form>
    </div>
  </main>

  <script>
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    // Fetch system info
    async function loadInfo() {
      try {
        const res = await fetch("/api/info");
        const data = await res.json();
        document.getElementById("doc-count").innerText = data.pdf_count;
        document.getElementById("model-name").innerText = "Groq: " + data.model;
        
        const list = document.getElementById("file-list");
        list.innerHTML = "";
        data.files.forEach(f => {
          const li = document.createElement("li");
          li.className = "file-item";
          li.innerHTML = '<span class="file-icon">📄</span> <span title="' + f + '">' + f + '</span>';
          list.appendChild(li);
        });
      } catch (e) {
        console.error("Failed to load info", e);
      }
    }
    loadInfo();

    function formatText(text) {
      if (window.marked) {
        return marked.parse(text);
      }
      return text.replace(/\\n/g, "<br>");
    }

    function addMessage(text, role, source) {
      const msgDiv = document.createElement("div");
      msgDiv.className = "message " + role;
      
      const avatar = document.createElement("div");
      avatar.className = "avatar " + role;
      avatar.innerText = role === "user" ? "You" : "ID";
      
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerHTML = formatText(text);

      if (source && role === "bot" && source !== "Greeting") {
        const tag = document.createElement("div");
        tag.className = "source-tag";
        tag.innerText = "Source: " + source;
        bubble.appendChild(tag);
      }

      msgDiv.appendChild(avatar);
      msgDiv.appendChild(bubble);
      chatMessages.appendChild(msgDiv);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
      const ind = document.createElement("div");
      ind.className = "message bot";
      ind.id = "typing-indicator";
      ind.innerHTML = `
        <div class="avatar bot">ID</div>
        <div class="bubble typing-indicator">
          <span></span><span></span><span></span>
        </div>
      `;
      chatMessages.appendChild(ind);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeTypingIndicator() {
      const ind = document.getElementById("typing-indicator");
      if (ind) ind.remove();
    }

    async function sendPrompt(text) {
      userInput.value = text;
      handleSubmit(new Event("submit"));
    }

    async function handleSubmit(e) {
      e.preventDefault();
      const query = userInput.value.trim();
      if (!query) return;

      addMessage(query, "user");
      userInput.value = "";
      userInput.disabled = true;
      sendBtn.disabled = true;
      showTypingIndicator();

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: query })
        });
        const data = await response.json();
        removeTypingIndicator();
        addMessage(data.answer, "bot", data.source);
      } catch (err) {
        removeTypingIndicator();
        addMessage("Error connecting to backend server: " + err.message, "bot");
      } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
      }
    }

    function clearChat() {
      chatMessages.innerHTML = `
        <div class="message bot">
          <div class="avatar bot">ID</div>
          <div class="bubble">
            <p>Chat cleared. Ask me any question about your technical manuals!</p>
          </div>
        </div>
      `;
    }
  </script>
</body>
</html>
"""

# Global retriever shared across HTTP requests
GLOBAL_RETRIEVER: LocalPdfRetriever | None = None
LOADED_FILES: list[str] = []


class RAGRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/api/info":
            info = {
                "pdf_count": len(LOADED_FILES),
                "chunk_count": len(GLOBAL_RETRIEVER.records) if GLOBAL_RETRIEVER else 0,
                "files": LOADED_FILES,
                "model": GROQ_MODEL,
                "tavily_enabled": bool(TAVILY_API_KEY),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(info).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                question = str(data.get("question", "")).strip()
            except Exception:
                question = ""

            if not question:
                res = {"answer": "Please provide a question.", "source": "None"}
            elif GLOBAL_RETRIEVER is None:
                res = {"answer": "Documents are not indexed yet.", "source": "Error"}
            else:
                res = query_rag(GLOBAL_RETRIEVER, question)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress routine HTTP GET logging to keep console clean
        return


def run_server() -> None:
    global GLOBAL_RETRIEVER, LOADED_FILES

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(DATA_DIR.rglob("*.pdf"))
    LOADED_FILES = [f.name for f in pdf_files]

    print("==================================================")
    print("IndusDocs Web UI — Technical RAG Assistant")
    print(f"Groq Model : {GROQ_MODEL} (Key configured: {bool(GROQ_API_KEY)})")
    print(f"Loaded PDFs: {len(LOADED_FILES)} files")
    print("Building local index...")
    records = load_pdf_chunks()
    GLOBAL_RETRIEVER = LocalPdfRetriever(records)
    print("Local index built successfully!")
    print(f"Server starting on http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server.")
    print("==================================================")

    # Open browser automatically in a separate thread
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), RAGRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping IndusDocs Web UI...")
        server.server_close()


if __name__ == "__main__":
    run_server()
