/**
 * Smart Passenger Counter - Enhanced Dashboard
 * Real-time WebSocket communication with Chart.js visualization
 * Displays passenger flow and occupancy trends over the last hour
 */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    MAX_DATA_POINTS: 60,        // Keep 60 data points (1 minute at 1 Hz)
    UPDATE_INTERVAL: 1000,      // Update every 1 second
    CHART_COLORS: {
        enter: '#00ff00',       // Green for entries
        exit: '#ff5252',        // Red for exits
        occupancy: '#00d4ff',   // Cyan for occupancy
        gridColor: 'rgba(0, 212, 255, 0.1)'
    }
};

// ============================================================================
// Global State
// ============================================================================

let state = {
    totalIn: 0,
    totalOut: 0,
    currentOccupancy: 0,
    detectionCount: 0,
    startTime: Date.now(),
    lastUpdateTime: Date.now(),
    frameCount: 0
};

let chartData = {
    timestamps: [],
    enterCounts: [],
    exitCounts: [],
    occupancyCounts: []
};

let charts = {
    enterExit: null,
    occupancy: null
};

// ============================================================================
// Initialize Charts
// ============================================================================

function initializeCharts() {
    // Chart.js configuration for Enter/Exit
    const ctxEnterExit = document.getElementById('chartEnterExit').getContext('2d');
    charts.enterExit = new Chart(ctxEnterExit, {
        type: 'line',
        data: {
            labels: chartData.timestamps,
            datasets: [
                {
                    label: 'Passengers In',
                    data: chartData.enterCounts,
                    borderColor: CONFIG.CHART_COLORS.enter,
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: CONFIG.CHART_COLORS.enter,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: 6
                },
                {
                    label: 'Passengers Out',
                    data: chartData.exitCounts,
                    borderColor: CONFIG.CHART_COLORS.exit,
                    backgroundColor: 'rgba(255, 82, 82, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: CONFIG.CHART_COLORS.exit,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' },
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: CONFIG.CHART_COLORS.gridColor,
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            return context.dataset.label + ': ' + context.parsed.y;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: CONFIG.CHART_COLORS.gridColor,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888888',
                        font: { size: 12 }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: CONFIG.CHART_COLORS.gridColor,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888888',
                        font: { size: 12 },
                        stepSize: 1
                    }
                }
            }
        }
    });

    // Chart.js configuration for Occupancy
    const ctxOccupancy = document.getElementById('chartOccupancy').getContext('2d');
    charts.occupancy = new Chart(ctxOccupancy, {
        type: 'line',
        data: {
            labels: chartData.timestamps,
            datasets: [
                {
                    label: 'Current Occupancy',
                    data: chartData.occupancyCounts,
                    borderColor: CONFIG.CHART_COLORS.occupancy,
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: CONFIG.CHART_COLORS.occupancy,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' },
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: CONFIG.CHART_COLORS.gridColor,
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            return context.dataset.label + ': ' + context.parsed.y + ' people';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: CONFIG.CHART_COLORS.gridColor,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888888',
                        font: { size: 12 }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: CONFIG.CHART_COLORS.gridColor,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888888',
                        font: { size: 12 },
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// ============================================================================
// Update Chart Data
// ============================================================================

function updateChartData(inCount, outCount, occupancy) {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
    });

    // Add new data point
    chartData.timestamps.push(timeString);
    chartData.enterCounts.push(inCount);
    chartData.exitCounts.push(outCount);
    chartData.occupancyCounts.push(occupancy);

    // Keep only the last MAX_DATA_POINTS
    if (chartData.timestamps.length > CONFIG.MAX_DATA_POINTS) {
        chartData.timestamps.shift();
        chartData.enterCounts.shift();
        chartData.exitCounts.shift();
        chartData.occupancyCounts.shift();
    }

    // Update charts
    if (charts.enterExit) {
        charts.enterExit.data.labels = chartData.timestamps;
        charts.enterExit.data.datasets[0].data = chartData.enterCounts;
        charts.enterExit.data.datasets[1].data = chartData.exitCounts;
        charts.enterExit.update('none'); // Update without animation for smooth real-time
    }

    if (charts.occupancy) {
        charts.occupancy.data.labels = chartData.timestamps;
        charts.occupancy.data.datasets[0].data = chartData.occupancyCounts;
        charts.occupancy.update('none');
    }
}

// ============================================================================
// Update UI Elements
// ============================================================================

function updateUI() {
    // Update stat cards
    document.getElementById('totalIn').textContent = state.totalIn;
    document.getElementById('totalOut').textContent = state.totalOut;
    document.getElementById('totalOccupancy').textContent = state.currentOccupancy;
    document.getElementById('detectionCount').textContent = state.detectionCount;

    // Update uptime
    const uptimeSeconds = Math.floor((Date.now() - state.startTime) / 1000);
    const hours = Math.floor(uptimeSeconds / 3600);
    const minutes = Math.floor((uptimeSeconds % 3600) / 60);
    const seconds = uptimeSeconds % 60;

    let uptimeString = '';
    if (hours > 0) uptimeString += `${hours}h `;
    if (minutes > 0) uptimeString += `${minutes}m `;
    uptimeString += `${seconds}s`;

    document.getElementById('uptime').textContent = uptimeString;
}

// ============================================================================
// WebSocket Connection
// ============================================================================

function initializeWebSocket() {
    const socket = io();

    // Connection established
    socket.on('connect', () => {
        console.log('✅ Connected to WebSocket server');
        updateConnectionStatus(true);
    });

    // Connection lost
    socket.on('disconnect', () => {
        console.warn('⚠️ Disconnected from WebSocket server');
        updateConnectionStatus(false);
    });

    // Receive real-time count updates
    socket.on('update_counts', (data) => {
        console.log('📊 Received count update:', data);

        // Handle both single door and multi-door configurations
        let totalIn = 0;
        let totalOut = 0;
        let totalOcc = 0;

        if (typeof data === 'object') {
            // Multi-door configuration
            for (let door in data) {
                if (data[door].enter !== undefined) {
                    totalIn += data[door].enter;
                }
                if (data[door].exit !== undefined) {
                    totalOut += data[door].exit;
                }
                if (data[door].occupancy !== undefined) {
                    totalOcc += data[door].occupancy;
                }
            }
        } else {
            // Single value configuration
            totalIn = data.IN || 0;
            totalOut = data.OUT || 0;
            totalOcc = data.total || 0;
        }

        // Update state
        state.totalIn = totalIn;
        state.totalOut = totalOut;
        state.currentOccupancy = totalOcc;
        state.lastUpdateTime = Date.now();

        // Update charts
        updateChartData(totalIn, totalOut, totalOcc);

        // Update UI
        updateUI();
    });

    // Receive detection updates
    socket.on('detection_update', (data) => {
        state.detectionCount = data.count || 0;
        updateUI();
    });

    // Receive FPS updates
    socket.on('fps_update', (data) => {
        const fps = data.fps || 0;
        document.getElementById('fpsValue').textContent = fps.toFixed(1);
    });

    // Error handling
    socket.on('error', (error) => {
        console.error('❌ WebSocket error:', error);
    });

    return socket;
}

function updateConnectionStatus(connected) {
    const statusBadge = document.querySelector('.status-badge');
    const statusIndicator = document.querySelector('.status-indicator');

    if (connected) {
        statusIndicator.style.background = '#00ff00';
        statusBadge.style.borderColor = '#00ff00';
        statusBadge.style.background = 'rgba(0, 255, 0, 0.2)';
    } else {
        statusIndicator.style.background = '#ff5252';
        statusBadge.style.borderColor = '#ff5252';
        statusBadge.style.background = 'rgba(255, 82, 82, 0.2)';
    }
}

// ============================================================================
// Control Functions
// ============================================================================

function resetCounters() {
    if (confirm('Are you sure you want to reset all counters? This action cannot be undone.')) {
        fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
            .then(response => response.json())
            .then(data => {
                console.log('✅ Counters reset:', data);
                state.totalIn = 0;
                state.totalOut = 0;
                state.currentOccupancy = 0;
                updateUI();

                // Clear chart data
                chartData.timestamps = [];
                chartData.enterCounts = [];
                chartData.exitCounts = [];
                chartData.occupancyCounts = [];

                if (charts.enterExit) charts.enterExit.update();
                if (charts.occupancy) charts.occupancy.update();

                showNotification('Counters have been reset', 'success');
            })
            .catch(error => {
                console.error('❌ Error resetting counters:', error);
                showNotification('Failed to reset counters', 'error');
            });
    }
}

function changeSource() {
    const source = document.getElementById('cameraSource').value;

    if (!source) {
        showNotification('Please select a camera source', 'warning');
        return;
    }

    fetch('/api/change_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: source })
    })
        .then(response => response.json())
        .then(data => {
            console.log('✅ Camera source changed:', data);
            showNotification(`Camera source changed to: ${source}`, 'success');
        })
        .catch(error => {
            console.error('❌ Error changing source:', error);
            showNotification('Failed to change camera source', 'error');
        });
}

// ============================================================================
// Notification System
// ============================================================================

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#00ff00' : type === 'error' ? '#ff5252' : '#00d4ff'};
        color: ${type === 'success' || type === 'error' ? '#000' : '#000'};
        border-radius: 8px;
        font-weight: bold;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ============================================================================
// FPS Counter
// ============================================================================

function initializeFPSCounter() {
    let lastFrameTime = performance.now();
    let frameCount = 0;
    let fps = 0;

    function updateFPS() {
        const now = performance.now();
        frameCount++;

        // Update FPS every 10 frames
        if (frameCount % 10 === 0) {
            fps = (10000 / (now - lastFrameTime)).toFixed(1);
            document.getElementById('fpsValue').textContent = fps;
            lastFrameTime = now;
        }

        requestAnimationFrame(updateFPS);
    }

    updateFPS();
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Smart Passenger Counter Dashboard...');

    // Initialize charts
    initializeCharts();
    console.log('✅ Charts initialized');

    // Initialize WebSocket
    initializeWebSocket();
    console.log('✅ WebSocket initialized');

    // Initialize FPS counter
    initializeFPSCounter();
    console.log('✅ FPS counter initialized');

    // Initial UI update
    updateUI();
    console.log('✅ Dashboard ready');

    // Add CSS animation for notifications
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(400px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        @keyframes slideOut {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(400px);
            }
        }
    `;
    document.head.appendChild(style);
});