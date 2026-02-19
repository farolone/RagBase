// Wissensdatenbank — Main App JS

// HTMX config
document.body.addEventListener('htmx:configRequest', (event) => {
    event.detail.headers['X-Requested-With'] = 'htmx';
});

// Toast notification system
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast-enter px-4 py-3 rounded-lg shadow-lg text-sm flex items-center gap-2 ${
        type === 'error'
            ? 'bg-red-600 text-white'
            : type === 'warning'
            ? 'bg-yellow-500 text-white'
            : 'bg-gray-800 text-white dark:bg-gray-200 dark:text-gray-900'
    }`;

    const icon = type === 'error' ? '!' : type === 'warning' ? '!' : '';
    toast.innerHTML = `${icon ? `<span class="font-bold">${icon}</span>` : ''}${message}`;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.remove('toast-enter');
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Simple markdown renderer for chat
function renderMarkdown(text) {
    if (!text) return '';
    return text
        // Code blocks
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Bold
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        // Citations [N] as badges
        .replace(/\[(\d+)\]/g, '<span class="inline-flex items-center justify-center w-5 h-5 text-xs bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full font-medium cursor-pointer">$1</span>')
        // Line breaks
        .replace(/\n/g, '<br>');
}
