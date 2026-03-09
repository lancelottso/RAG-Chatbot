const chatMessages = document.getElementById("chatMessages");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");

// Auto-resize textarea
queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

// Send on Enter (Shift+Enter for newline)
queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

sendBtn.addEventListener("click", handleSend);

async function handleSend() {
    const query = queryInput.value.trim();
    if (!query) return;

    appendMessage(query, "user");
    queryInput.value = "";
    queryInput.style.height = "auto";
    setLoading(true);

    const typingEl = appendTypingIndicator();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });
        const data = await res.json();
        typingEl.remove();

        if (data.error) {
            appendMessage(data.error, "bot", [], true);
        } else {
            appendMessage(data.answer, "bot", data.sources || []);
        }
    } catch (err) {
        typingEl.remove();
        appendMessage("Sorry, I couldn't reach the server. Please try again.", "bot", [], true);
    } finally {
        setLoading(false);
    }
}

function appendMessage(text, role, sources = [], isError = false) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}-message`;

    const avatar = document.createElement("div");
    avatar.className = `message-avatar ${role}-avatar`;
    avatar.innerHTML = role === "bot"
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;

    const content = document.createElement("div");
    content.className = "message-content";
    if (isError) content.style.borderColor = "rgba(248,113,113,.4)";

    const formatted = text.split(/\n{2,}/).map(p => `<p>${escapeHtml(p).replace(/\n/g, "<br>")}</p>`).join("");
    content.innerHTML = formatted;

    if (sources.length > 0) {
        const details = document.createElement("details");
        details.className = "sources-list";
        details.innerHTML = `<summary>Sources (${sources.length})</summary>` +
            sources.map(s => `<a href="${escapeHtml(s.url)}" target="_blank">${escapeHtml(s.title || s.url)}</a>`).join("");
        content.appendChild(details);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(content);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function appendTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "message bot-message";
    wrapper.innerHTML = `
        <div class="message-avatar bot-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
        </div>
        <div class="message-content">
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>`;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
}

function setLoading(isLoading) {
    sendBtn.disabled = isLoading;
    queryInput.disabled = isLoading;
    if (!isLoading) queryInput.focus();
}

function scrollToBottom() {
    document.querySelector(".chat-container").scrollTop = document.querySelector(".chat-container").scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}