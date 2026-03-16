import { useState, useRef, useEffect } from "react";

// ─── Typing Indicator ────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex items-end gap-3 mb-4">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-white text-xs font-bold shrink-0 shadow-md">
        AI
      </div>
      <div className="bg-white border border-slate-100 shadow-sm rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1.5">
        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
      </div>
    </div>
  );
}

// ─── Message Bubble ───────────────────────────────────────────────────────────
function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex items-end gap-3 mb-4 ${
        isUser ? "flex-row-reverse" : "flex-row"
      }`}
    >
      {/* Avatar */}
      {isUser ? (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-600 to-slate-800 flex items-center justify-center text-white text-xs font-bold shrink-0 shadow-md">
          You
        </div>
      ) : (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-white text-xs font-bold shrink-0 shadow-md">
          AI
        </div>
      )}

      {/* Bubble */}
      <div
        className={`max-w-[75%] px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser
            ? "bg-gradient-to-br from-violet-600 to-indigo-600 text-white rounded-2xl rounded-br-sm"
            : "bg-white border border-slate-100 text-slate-800 rounded-2xl rounded-bl-sm"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}

// ─── Main Chatbot Component ───────────────────────────────────────────────────
export default function Chatbot() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      content: "Hey there! 👋 I'm your AI assistant. How can I help you today?",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isOpen, setIsOpen] = useState(true);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // ── Replace this function with your real API call ──────────────────────────
  async function fetchBotReply(userMessage, conversationHistory) {
    // TODO: Integrate your API here.
    // `userMessage`        – the latest message string
    // `conversationHistory` – full messages array for context
    //
    // Example shape for most chat APIs:
    // const response = await fetch("/api/chat", {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify({ message: userMessage, history: conversationHistory }),
    // });
    // const data = await response.json();
    // return data.reply;

    // ── Stub reply (remove once you wire up your API) ──
    await new Promise((res) => setTimeout(res, 1500));
    return `This is a placeholder reply to: "${userMessage}". Wire up your API in the fetchBotReply function.`;
  }
  // ───────────────────────────────────────────────────────────────────────────

  async function handleSend() {
    const text = inputValue.trim();
    if (!text || isTyping) return;

    const userMessage = {
      id: Date.now(),
      role: "user",
      content: text,
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputValue("");
    setIsTyping(true);

    try {
      const reply = await fetchBotReply(text, updatedMessages);
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, role: "assistant", content: reply },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: "⚠️ Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleClear() {
    setMessages([
      {
        id: Date.now(),
        role: "assistant",
        content: "Chat cleared. How can I help you?",
      },
    ]);
  }

  return (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center p-4 font-sans">
      {/* Chat Window */}
      {isOpen && (
        <div
          className="w-full max-w-lg bg-slate-50 rounded-3xl shadow-2xl flex flex-col overflow-hidden border border-slate-200"
          style={{ height: "600px" }}
        >
          {/* ── Header ─────────────────────────────────────────────────────── */}
          <div className="bg-gradient-to-r from-violet-600 to-indigo-600 px-5 py-4 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white/20 backdrop-blur flex items-center justify-center text-white font-bold text-sm shadow-inner">
                AI
              </div>
              <div>
                <p className="text-white font-semibold text-sm leading-tight">
                  AI Assistant
                </p>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-violet-200 text-xs">
                    {isTyping ? "Typing..." : "Online"}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleClear}
                title="Clear chat"
                className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
              <button
                onClick={() => setIsOpen(false)}
                title="Close"
                className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>

          {/* ── Messages ───────────────────────────────────────────────────── */}
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-0 scroll-smooth">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {isTyping && <TypingIndicator />}

            <div ref={messagesEndRef} />
          </div>

          {/* ── Input Bar ──────────────────────────────────────────────────── */}
          <div className="px-4 py-4 bg-white border-t border-slate-100 shrink-0">
            <div className="flex items-end gap-3 bg-slate-100 rounded-2xl px-4 py-2.5">
              <textarea
                ref={inputRef}
                rows={1}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message… (Enter to send)"
                disabled={isTyping}
                className="flex-1 bg-transparent resize-none outline-none text-sm text-slate-800 placeholder-slate-400 max-h-32 leading-relaxed disabled:opacity-50"
                style={{ minHeight: "24px" }}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height = e.target.scrollHeight + "px";
                }}
              />
              <button
                onClick={handleSend}
                disabled={!inputValue.trim() || isTyping}
                className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center text-white shadow-md hover:shadow-lg hover:scale-105 active:scale-95 transition-all disabled:opacity-40 disabled:scale-100 disabled:shadow-none shrink-0"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-4 h-4 translate-x-px"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
                  />
                </svg>
              </button>
            </div>
            <p className="text-center text-slate-400 text-[10px] mt-2">
              Shift+Enter for new line · Enter to send
            </p>
          </div>
        </div>
      )}

      {/* Floating Re-open Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center text-white shadow-2xl hover:scale-110 active:scale-95 transition-transform"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-7 h-7"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
        </button>
      )}
    </div>
  );
}
