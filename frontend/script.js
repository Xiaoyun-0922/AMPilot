// AMPilot Frontend JavaScript
class AMPilotChat {
    constructor() {
        console.log('🚀 AMPilotChat constructor started');
        this.apiBaseUrl = 'http://127.0.0.1:8002';
        this.sessionId = this.generateSessionId();
        this.isConnected = false;
        this.websocket = null;
        this.logoAvailable = false;
        // Streaming coordination flags to avoid duplicate bubbles
        this.ignoreNextFinalResponse = false;

        console.log('📋 Session ID:', this.sessionId);

        this.initializeElements();
        // Cache initial HTML (for quick reset)
        this.initialMessagesHTML = this.chatMessages.innerHTML;

        this.setupEventListeners();
        this.initializeGlobalFunctions();
        this.preloadLogo('/static/logo.png');
        this.ensureTypingIndicatorAtEnd();
        this.connectWebSocket();

        console.log('✅ AMPilotChat constructor completed');
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
    }

    setupEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Enter key to send message
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.chatInput.addEventListener('input', () => {
            this.chatInput.style.height = 'auto';
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
        });

        // New conversation
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) newChatBtn.addEventListener('click', () => this.newConversation());
    }

    connectWebSocket() {
        try {
            const wsUrl = `ws://127.0.0.1:8002/ws/${this.sessionId}`;
            console.log('🔌 Attempting WebSocket connection to:', wsUrl);
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected successfully');
                this.isConnected = true;
                this.updateConnectionStatus(true);
            };

            this.websocket.onmessage = (event) => {
                console.log('📥 WebSocket message received:', event.data);
                const data = JSON.parse(event.data);
                console.log('📋 Parsed data:', data);
                this.handleWebSocketMessage(data);
            };

            this.websocket.onclose = (event) => {
                console.log('❌ WebSocket disconnected. Code:', event.code, 'Reason:', event.reason);
                this.isConnected = false;
                this.updateConnectionStatus(false);
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };

            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.isConnected = false;
                this.updateConnectionStatus(false);
            };

        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.isConnected = false;
            this.updateConnectionStatus(false);
        }
    }

    updateConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-indicator span');
        if (connected) {
            statusDot.style.background = '#10b981';
            statusText.textContent = 'Online';
        } else {
            statusDot.style.background = '#ef4444';
            statusText.textContent = 'Offline';
        }
    }

    preloadLogo(url) {
        const img = new Image();
        img.onload = () => { this.logoAvailable = true; };
        img.onerror = () => { this.logoAvailable = false; };
        img.src = url;
    }

    ensureTypingIndicatorAtEnd() {
        if (this.typingIndicator && this.typingIndicator.parentNode) {
            this.chatMessages.appendChild(this.typingIndicator);
        }
    }

    async newConversation() {
        try {
            // Delete old session on server (best-effort)
            await fetch(`${this.apiBaseUrl}/sessions/${this.sessionId}`, { method: 'DELETE' });
        } catch (e) { /* ignore */ }
        // Reset session id and reconnect WS
        this.sessionId = this.generateSessionId();
        try { if (this.websocket) this.websocket.close(); } catch (e) {}
        // Reset UI to initial state
        this.chatMessages.innerHTML = this.initialMessagesHTML;
        // Re-select typing indicator after DOM reset
        this.typingIndicator = document.getElementById('typingIndicator');
        this.ensureTypingIndicatorAtEnd();
        this.chatInput.value = '';
        this.chatInput.disabled = false;
        this.sendButton.disabled = false;
        this.connectWebSocket();
        this.scrollToBottom();
    }

    addCopyButton(containerEl, getText) {
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.title = 'Copy';
        btn.innerHTML = '<i class="fas fa-copy"></i>';
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                await navigator.clipboard.writeText(getText());
                btn.classList.add('copied');
                setTimeout(() => btn.classList.remove('copied'), 800);
            } catch (err) {
                console.warn('Clipboard failed, falling back');
                const ta = document.createElement('textarea');
                ta.value = getText();
                document.body.appendChild(ta);
                ta.select();
                try { document.execCommand('copy'); } catch (e2) {}
                document.body.removeChild(ta);
            }
        });
        containerEl.appendChild(btn);
        return btn;
    }


    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'typing':
                this.showTypingIndicator();
                break;
            case 'stream_start':
                this.hideTypingIndicator();
                this.ignoreNextFinalResponse = false; // starting new stream
                this.currentStreamingMessage = this.addStreamingMessage('assistant');
                break;
            case 'stream_chunk':
                if (this.currentStreamingMessage) {
                    this.appendToStreamingMessage(data.content);
                    // Auto-scroll to bottom during streaming for better UX
                    this.scrollToBottom();
                }
                break;
            case 'stream_end':
                if (this.currentStreamingMessage) {
                    this.finalizeStreamingMessage(data.timestamp);
                    this.currentStreamingMessage = null;
                }
                // Mark to ignore the next final_response to prevent duplicate bubbles
                this.ignoreNextFinalResponse = true;
                // Re-enable input after streaming ends
                this.chatInput.disabled = false;
                this.sendButton.disabled = false;
                this.chatInput.focus();
                break;
            case 'final_response':
                this.hideTypingIndicator();
                if (this.ignoreNextFinalResponse) {
                    // We already rendered the streamed message as the final bubble
                    this.ignoreNextFinalResponse = false; // reset for next turn
                    break;
                }
                this.addMessage(data.content, 'assistant', data.timestamp);
                // Re-enable input after final response
                this.chatInput.disabled = false;
                this.sendButton.disabled = false;
                this.chatInput.focus();
                break;
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        // Clear input and disable controls during processing
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';
        this.chatInput.disabled = true;
        this.sendButton.disabled = true;

        // Hide welcome block on first message
        const welcome = document.querySelector('.welcome-message');
        if (welcome) welcome.style.display = 'none';

        // Add user message to chat
        this.addMessage(message, 'user');

        // Send via WebSocket if connected, otherwise use REST API
        console.log('🔍 Connection status - isConnected:', this.isConnected, 'readyState:', this.websocket?.readyState);
        if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
            console.log('📤 Sending via WebSocket:', message);
            this.websocket.send(JSON.stringify({ message: message }));
        } else {
            console.log('📤 Sending via REST API:', message);
            await this.sendMessageViaAPI(message);
        }

        // Re-enable controls (will be handled by message response)
        // this.sendButton.disabled = false;
    }

    async sendMessageViaAPI(message) {
        this.showTypingIndicator();

        try {
            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.hideTypingIndicator();
            this.addMessage(data.response, 'assistant', data.timestamp);

        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessage(
                'Sorry, I encountered an error while processing your request. Please try again.',
                'assistant'
            );
        }
    }

    addStreamingMessage(sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (sender === 'assistant' && this.logoAvailable) {
            avatar.innerHTML = '<img src="/static/logo.png" alt="logo" />';
        } else {
            avatar.textContent = sender === 'user' ? 'U' : 'A';
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<div class="streaming-content"></div>';

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);

        this.chatMessages.appendChild(messageDiv);
        this.ensureTypingIndicatorAtEnd();
        this.scrollToBottom();

        return {
            messageDiv,
            contentDiv: contentDiv.querySelector('.streaming-content'),
            timeDiv
        };
    }

    appendToStreamingMessage(htmlContent) {
        if (this.currentStreamingMessage) {
            // Accumulate the raw content and parse markdown
            this.currentStreamingMessage.rawContent = (this.currentStreamingMessage.rawContent || '') + htmlContent;

            // Add streaming cursor class for visual feedback
            this.currentStreamingMessage.contentDiv.classList.add('streaming-content');

            // Parse markdown and update content
            this.currentStreamingMessage.contentDiv.innerHTML = this.parseMarkdown(this.currentStreamingMessage.rawContent);

            // Smooth scroll to bottom during streaming
            this.scrollToBottom();
        }
    }

    finalizeStreamingMessage(timestamp) {
        if (this.currentStreamingMessage) {
            if (timestamp) {
                this.currentStreamingMessage.timeDiv.textContent = new Date(timestamp).toLocaleTimeString();
            }

            // Remove streaming cursor class
            this.currentStreamingMessage.contentDiv.classList.remove('streaming-content');

            // Render MathJax for the final streaming message
            this.renderMathJax(this.currentStreamingMessage.contentDiv);
            // Add copy button for the finalized streaming message
            this.addCopyButton(this.currentStreamingMessage.contentDiv, () => this.currentStreamingMessage.contentDiv.innerText || '');
        }
    }

    addMessage(content, sender, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (sender === 'assistant' && this.logoAvailable) {
            avatar.innerHTML = '<img src="/static/logo.png" alt="logo" />';
        } else {
            avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        }

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        // Handle different content types
        if (sender === 'assistant') {
            if (this.isJsonResponse(content)) {
                messageContent.innerHTML = this.formatJsonResponse(content);
            } else if (content.includes('<') && content.includes('>')) {
                // Support HTML content for rich formatting
                messageContent.innerHTML = content;
            } else {
                // Parse markdown for assistant messages
                messageContent.innerHTML = this.parseMarkdown(content);
            }
        } else {
            messageContent.textContent = content;
        }

        // Add timestamp
        if (timestamp) {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = new Date(timestamp).toLocaleTimeString();
            messageContent.appendChild(timeDiv);
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        // Always append at the end and keep typing indicator as the last element
        this.chatMessages.appendChild(messageDiv);
        this.ensureTypingIndicatorAtEnd();
        this.scrollToBottom();

        // Add copy button for assistant messages
        if (sender === 'assistant') {
            this.addCopyButton(messageContent, () => messageContent.innerText || '');
            // Render MathJax for mathematical formulas
            this.renderMathJax(messageContent);
        }
    }

    isJsonResponse(content) {
        try {
            const parsed = JSON.parse(content);
            return parsed && typeof parsed === 'object' && parsed.status;
        } catch {
            return false;
        }
    }

    parseMarkdown(text) {
        if (!text) return '';

        try {
            // Configure marked options
            marked.setOptions({
                breaks: true,        // Convert \n to <br>
                gfm: true,          // GitHub Flavored Markdown
                tables: true,       // Support tables
                sanitize: false,    // Allow HTML (we trust our backend)
                highlight: function(code, lang) {
                    // Syntax highlighting for code blocks
                    if (lang && hljs.getLanguage(lang)) {
                        try {
                            return hljs.highlight(code, { language: lang }).value;
                        } catch (err) {
                            console.warn('Highlight.js error:', err);
                        }
                    }
                    return hljs.highlightAuto(code).value;
                }
            });

            // Parse markdown to HTML
            let html = marked.parse(text);

            // Post-process to add copy buttons to code blocks
            html = this.addCodeBlockCopyButtons(html);

            return html;
        } catch (error) {
            console.error('Markdown parsing error:', error);
            // Fallback to simple text with line breaks
            return text.replace(/\n/g, '<br>');
        }
    }

    // Render MathJax after content is added to DOM
    renderMathJax(element) {
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([element]).catch((err) => {
                console.warn('MathJax rendering error:', err);
            });
        }
    }

    addCodeBlockCopyButtons(html) {
        // Add copy buttons to code blocks
        return html.replace(/<pre><code([^>]*)>([\s\S]*?)<\/code><\/pre>/g, (match, attrs, code) => {
            const cleanCode = code.replace(/<[^>]*>/g, ''); // Remove HTML tags for copying
            const escapedCode = this.escapeHtml(cleanCode);
            return `<div class="code-block-container" style="position: relative;">
                <pre><code${attrs}>${code}</code></pre>
                <button class="code-copy-btn" onclick="window.copyCodeBlock('${escapedCode}')"
                        style="position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,0.1);
                               border: none; border-radius: 4px; padding: 4px 8px; cursor: pointer;
                               font-size: 12px; color: #666;" title="Copy code">
                    <i class="fas fa-copy"></i>
                </button>
            </div>`;
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/'/g, '&#39;');
    }

    // Initialize global functions
    initializeGlobalFunctions() {
        // Global function for code block copy buttons
        window.copyCodeBlock = (code) => {
            const decodedCode = code.replace(/&#39;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
            navigator.clipboard.writeText(decodedCode).then(() => {
                console.log('Code copied to clipboard');
                // Could add visual feedback here
            }).catch(err => {
                console.error('Failed to copy code:', err);
            });
        };
    }

    formatJsonResponse(jsonContent) {
        try {
            const data = JSON.parse(jsonContent);
            let html = `<div class="json-response">`;

            // Summary
            if (data.summary) {
                html += `<div class="summary"><strong>Summary:</strong> ${data.summary}</div>`;
            }

            // Peptides found
            if (data.retrieved_peptides && data.retrieved_peptides.length > 0) {
                html += `<div class="peptides-section">`;
                html += `<h4>Found ${data.retrieved_peptides.length} Peptide(s):</h4>`;

                data.retrieved_peptides.forEach((peptide, index) => {
                    html += `<div class="peptide-card">`;
                    html += `<div class="peptide-header">`;
                    html += `<strong>${peptide.name || 'Unknown'}</strong>`;
                    if (peptide.dramp_id) {
                        html += ` <span class="dramp-id">(${peptide.dramp_id})</span>`;
                    }
                    html += `</div>`;

                    if (peptide.sequence) {
                        html += `<div class="sequence"><strong>Sequence:</strong> <code>${peptide.sequence}</code></div>`;
                    }

                    if (peptide.description) {
                        html += `<div class="description">${peptide.description}</div>`;
                    }

                    if (peptide.reference) {
                        html += `<div class="reference"><em>Reference: ${peptide.reference}</em></div>`;
                    }

                    html += `</div>`;
                });
                html += `</div>`;
            }

            // Scientific insights
            if (data.scientific_insights) {
                html += `<div class="insights"><strong>Insights:</strong> ${data.scientific_insights}</div>`;
            }

            // Recommendations
            if (data.recommendations) {
                html += `<div class="recommendations"><strong>Recommendations:</strong> ${data.recommendations}</div>`;
            }

            html += `</div>`;

            // Add CSS for JSON response formatting
            if (!document.getElementById('json-response-styles')) {
                const style = document.createElement('style');
                style.id = 'json-response-styles';
                style.textContent = `
                    .json-response { font-size: 13px; }
                    .summary { margin-bottom: 15px; padding: 10px; background: #f0f9ff; border-radius: 8px; }
                    .peptides-section h4 { margin: 15px 0 10px 0; color: #4f46e5; }
                    .peptide-card { margin: 10px 0; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; }
                    .peptide-header { font-weight: 600; margin-bottom: 8px; }
                    .dramp-id { color: #6b7280; font-weight: normal; }
                    .sequence { margin: 8px 0; }
                    .sequence code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
                    .description { margin: 8px 0; line-height: 1.4; }
                    .reference { margin-top: 8px; font-size: 12px; color: #6b7280; }
                    .insights, .recommendations { margin-top: 15px; padding: 10px; background: #f0fdf4; border-radius: 8px; }
                `;
                document.head.appendChild(style);
            }

            return html;
        } catch (error) {
            return jsonContent; // Fallback to plain text
        }
    }

    showTypingIndicator() {
        if (this.logoAvailable) {
            const avatar = this.typingIndicator.querySelector('.message-avatar');
            if (avatar) avatar.innerHTML = '<img src="/static/logo.png" alt="logo" />';
        }
        this.typingIndicator.style.display = 'flex';
        this.ensureTypingIndicatorAtEnd();
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
}

// Example question function
function sendExampleQuestion(question) {
    const chatInput = document.getElementById('chatInput');
    chatInput.value = question;
    chatInput.focus();

    // Trigger send
    if (window.ampilotChat) {
        window.ampilotChat.sendMessage();
    }
}

// Test WebSocket function
function testWebSocket() {
    console.log('🧪 Test WebSocket button clicked');
    if (window.ampilotChat) {
        console.log('📋 Chat instance exists');
        console.log('🔍 Connection status:', window.ampilotChat.isConnected);
        console.log('🔍 WebSocket state:', window.ampilotChat.websocket?.readyState);

        // Set test message and send
        const chatInput = document.getElementById('chatInput');
        chatInput.value = 'hello test';
        window.ampilotChat.sendMessage();
    } else {
        console.error('❌ Chat instance not found');
    }
}

// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌟 DOM Content Loaded - Initializing AMPilot Chat');
    try {
        window.ampilotChat = new AMPilotChat();
        console.log('🎉 AMPilot Chat initialized successfully');
    } catch (error) {
        console.error('💥 Failed to initialize AMPilot Chat:', error);
    }
});
