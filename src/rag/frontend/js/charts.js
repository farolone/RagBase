// Chart.js dashboard charts

const PLATFORM_COLORS = {
    web: '#3B82F6',
    pdf: '#EF4444',
    youtube: '#F43F5E',
    reddit: '#F97316',
    twitter: '#0EA5E9',
};

function renderPlatformChart(canvasId, platforms) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const labels = Object.keys(platforms);
    const values = Object.values(platforms);
    const colors = labels.map(l => PLATFORM_COLORS[l] || '#6B7280');

    // Check if dark mode
    const isDark = document.documentElement.classList.contains('dark');
    const textColor = isDark ? '#9CA3AF' : '#6B7280';

    new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: textColor, padding: 16 },
                },
            },
        },
    });
}

function renderTimelineChart(canvasId, timeline) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const isDark = document.documentElement.classList.contains('dark');
    const textColor = isDark ? '#9CA3AF' : '#6B7280';
    const gridColor = isDark ? '#374151' : '#E5E7EB';

    const labels = timeline.map(t => t.date);
    const values = timeline.map(t => t.count);

    new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Dokumente',
                data: values,
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointBackgroundColor: '#3B82F6',
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: textColor, stepSize: 1 },
                    grid: { color: gridColor },
                },
                x: {
                    ticks: { color: textColor },
                    grid: { color: gridColor },
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
}
