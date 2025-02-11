document.addEventListener("DOMContentLoaded", function () {
    var ws = new WebSocket("ws://localhost:8000/ws");

    // HTML Elements
    var totalDetections = document.getElementById("totalDetections");
    var totalViolations = document.getElementById("totalViolations");
    var mostFrequentClass = document.getElementById("mostFrequentClass");
    var dataTable = document.getElementById("dataTable");

    // Violation counts by time of day
    var violationCountsByTime = {
        morning: 0,
        afternoon: 0,
        night: 0
    };

    // Data Storage
    var detectionCounts = {}; // { "person": 5, "car": 3, ... }
    var confidenceData = {};  // { "person": [0.9, 0.8], "car": [0.7] }

    // Bar Chart Initialization
    var barCtx = document.getElementById("barChart").getContext("2d");
    var barChart = new Chart(barCtx, {
        type: "bar",
        data: {
            labels: [],
            datasets: [{
                label: "Average Confidence (%)",
                backgroundColor: [], // Colors will be assigned dynamically
                data: [],
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
            }
        }
    });

    // Pie Chart Initialization
    var pieCtx = document.getElementById('violationPieChart').getContext('2d');
    var violationPieChart = new Chart(pieCtx, {
        type: 'pie',
        data: {
            labels: ['Morning (6 AM - 12 PM)', 'Afternoon (12 PM - 6 PM)', 'Night (6 PM - 12 AM)'],
            datasets: [{
                data: [0, 0, 0], // Initial values
                backgroundColor: ['#00f2ff', '#ff8c00', '#ff2d00'], 
                borderColor: '#121212',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#fff',
                        font: { weight: 'bold' }
                    }
                }
            }
        }
    });

    ws.onmessage = function (event) {
        var data = JSON.parse(event.data);

        // Update summary cards
        totalDetections.innerHTML = data.summary.total_detections;
        totalViolations.innerHTML = data.summary.total_violations;
        mostFrequentClass.innerHTML = data.summary.most_frequent_class;

        // Process each detection
        data.timestamp.forEach(function(timestamp, index) {
            var detectedClass = data.class[index];
            var confidence = data.confidence[index];

            // Update detection counts and confidence data
            if (!detectionCounts[detectedClass]) {
                detectionCounts[detectedClass] = 0;
                confidenceData[detectedClass] = [];
            }
            detectionCounts[detectedClass] += 1;
            confidenceData[detectedClass].push(confidence);

            // Update violation count if violation occurred
            if (data.restricted_area_violation[index]) {
                updateViolationCounts(timestamp);
            }

            // Add new row to the table
            var newRow = `<tr>
                <td>${timestamp}</td>
                <td>${detectedClass}</td>
                <td>${(confidence).toFixed(2)}%</td>
                <td>${data.restricted_area_violation[index]}</td>
            </tr>`;
            dataTable.innerHTML = newRow + dataTable.innerHTML;
        });

        // Update the bar chart
        updateBarChart();
    };

    function updateBarChart() {
        barChart.data.labels = Object.keys(detectionCounts);
        
        barChart.data.datasets[0].data = Object.keys(confidenceData).map(cls => {
            let sum = confidenceData[cls].reduce((a, b) => a + b, 0);
            return (sum / confidenceData[cls].length);
        });

        // Assign different colors dynamically
        let colors = [
            "#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#FFD700", "#00FFFF", 
            "#8A2BE2", "#DC143C", "#20B2AA", "#FF4500"
        ]; // More colors added for variety

        barChart.data.datasets[0].backgroundColor = barChart.data.labels.map((_, i) => colors[i % colors.length]);

        barChart.update();
    }

    function updateViolationCounts(timestamp) {
        var violationTime = new Date(timestamp).getHours(); 

        if (violationTime >= 6 && violationTime < 12) {
            violationCountsByTime.morning += 1;
        } else if (violationTime >= 12 && violationTime < 18) {
            violationCountsByTime.afternoon += 1;
        } else {
            violationCountsByTime.night += 1;
        }

        // Update Pie Chart
        violationPieChart.data.datasets[0].data = [
            violationCountsByTime.morning,
            violationCountsByTime.afternoon,
            violationCountsByTime.night
        ];
        violationPieChart.update();
    }

    // Simulated data for testing (REMOVE IN LIVE SYSTEM)
    // setInterval(() => {
    //     const fakeTimestamp = new Date().toISOString();
    //     updateViolationCounts(fakeTimestamp);
    // }, 5000);
});
