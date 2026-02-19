// Chat SSE handling

function chatApp() {
    return {
        sessions: [],
        currentSession: null,
        messages: [],
        question: '',
        streaming: false,
        showSource: null,

        async init() {
            await this.loadSessions();
        },

        async loadSessions() {
            try {
                const resp = await fetch('/api/chat/sessions');
                const data = await resp.json();
                this.sessions = data.sessions;
            } catch (e) {}
        },

        async newSession() {
            try {
                const resp = await fetch('/api/chat/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: 'Neuer Chat' }),
                });
                const data = await resp.json();
                this.currentSession = data.id;
                this.messages = [];
                await this.loadSessions();
            } catch (e) { showToast('Fehler', 'error'); }
        },

        async loadSession(sessionId) {
            this.currentSession = sessionId;
            try {
                const resp = await fetch(`/api/chat/sessions/${sessionId}/messages`);
                const data = await resp.json();
                this.messages = data.messages;
                this.scrollToBottom();
            } catch (e) {}
        },

        async deleteSession(sessionId) {
            await fetch(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' });
            if (this.currentSession === sessionId) {
                this.currentSession = null;
                this.messages = [];
            }
            await this.loadSessions();
        },

        async ask() {
            const q = this.question.trim();
            if (!q || this.streaming) return;

            // Create session if needed
            if (!this.currentSession) {
                await this.newSession();
            }

            // Add user message
            this.messages.push({ role: 'user', content: q });
            this.question = '';
            this.streaming = true;
            this.scrollToBottom();

            // Start SSE stream
            let sources = [];
            let fullAnswer = '';

            try {
                const resp = await fetch('/api/ask/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: q,
                        session_id: this.currentSession,
                    }),
                });

                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                // Add empty assistant message
                this.messages.push({ role: 'assistant', content: '', sources: [] });
                const msgIdx = this.messages.length - 1;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Keep incomplete line in buffer

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.type === 'sources') {
                                sources = data.sources;
                                this.messages[msgIdx].sources = sources;
                            } else if (data.type === 'content') {
                                fullAnswer += data.content;
                                this.messages[msgIdx].content = fullAnswer;
                                this.scrollToBottom();
                            } else if (data.type === 'done') {
                                // Stream complete
                            }
                        } catch (e) {}
                    }
                }
            } catch (e) {
                // Fallback to non-streaming
                try {
                    const resp = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: q }),
                    });
                    const data = await resp.json();
                    const msgIdx = this.messages.length - 1;
                    if (this.messages[msgIdx].role === 'assistant') {
                        this.messages[msgIdx].content = data.answer;
                        this.messages[msgIdx].sources = data.sources;
                    } else {
                        this.messages.push({ role: 'assistant', content: data.answer, sources: data.sources });
                    }
                } catch (e2) {
                    showToast('Fehler bei der Antwort', 'error');
                }
            }

            this.streaming = false;
            this.scrollToBottom();

            // Update session title with first question
            if (this.sessions.length > 0) {
                const session = this.sessions.find(s => s.id === this.currentSession);
                if (session && session.title === 'Neuer Chat') {
                    session.title = q.substring(0, 50) + (q.length > 50 ? '...' : '');
                }
            }
        },

        async sendFeedback(msg, rating) {
            try {
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: this.currentSession,
                        question: '',
                        answer: msg.content,
                        rating: rating,
                    }),
                });
                showToast(rating > 0 ? 'Danke für das Feedback!' : 'Feedback gespeichert');
            } catch (e) {}
        },

        scrollToBottom() {
            this.$nextTick(() => {
                const el = this.$refs.messages;
                if (el) el.scrollTop = el.scrollHeight;
            });
        }
    };
}
