const socket = io();

// ------------------------------
// Realtime Charts
// ------------------------------
let enterExitChart;
let occupancyChart;

let timeLabels = [];
let enterData = [];
let exitData = [];
let occupancyData = [];

// ------------------------------
// Create Charts
// ------------------------------
function createCharts() {
    const ctx1 = document.getElementById("chartEnterExit").getContext("2d");
    const ctx2 = document.getElementById("chartOccupancy").getContext("2d");

    enterExitChart = new Chart(ctx1, {
        type: "line",
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: "Enter",
                    borderColor: "#4caf50",
                    data: enterData,
                    tension: 0.25
                },
                {
                    label: "Exit",
                    borderColor: "#ff5252",
                    data: exitData,
                    tension: 0.25
                }
            ]
        },
        options: { responsive: true }
    });

    occupancyChart = new Chart(ctx2, {
        type: "line",
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: "Occupancy",
                    borderColor: "#2979ff",
                    data: occupancyData,
                    tension: 0.25
                }
            ]
        },
        options: { responsive: true }
    });
}

createCharts();

// ------------------------------
// WebSocket Handler
// ------------------------------
socket.on("update_counts", (doorStats) => {
    const container = document.getElementById("door-stats");
    container.innerHTML = "";

    let totalEnter = 0, totalExit = 0, totalOcc = 0;

    for (let door in doorStats) {
        const stats = doorStats[door];

        container.innerHTML += `
            <div class="door-card">
                <h3>${door.toUpperCase()}</h3>
                Enter: ${stats.enter}<br>
                Exit: ${stats.exit}<br>
                Occupancy: ${stats.occupancy}
            </div>
        `;

        totalEnter += stats.enter;
        totalExit += stats.exit;
        totalOcc += stats.occupancy;
    }

    // Update charts
    if (timeLabels.length > 50) {
        timeLabels.shift();
        enterData.shift();
        exitData.shift();
        occupancyData.shift();
    }

    timeLabels.push(new Date().toLocaleTimeString());
    enterData.push(totalEnter);
    exitData.push(totalExit);
    occupancyData.push(totalOcc);

    enterExitChart.update();
    occupancyChart.update();
});

// ------------------------------
// FPS Counter
// ------------------------------
let lastFrameTime = performance.now();

function updateFPS() {
    const now = performance.now();
    const fps = (1000 / (now - lastFrameTime)).toFixed(1);
    lastFrameTime = now;

    document.getElementById("fps").innerText = `FPS: ${fps}`;
    requestAnimationFrame(updateFPS);
}
updateFPS();

// ------------------------------
// API: Reset Counters
// ------------------------------
function resetCounters() {
    fetch("/api/reset", { method: "POST" })
        .then(r => r.json())
        .then(() => alert("Counters reset"));
}

// ------------------------------
// API: Change Video Source
// ------------------------------
function changeSource() {
    const src = document.getElementById("cameraSource").value;

    fetch("/api/change_source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: src })
    }).then(r => r.json())
        .then(() => alert("Camera source changed"));
}
