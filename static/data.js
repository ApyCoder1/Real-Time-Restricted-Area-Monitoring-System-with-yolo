
        const ws = new WebSocket("ws://localhost:8000/ws/data");

        // WebSocket Event Listeners
        ws.onopen = function() {
            console.log("Connected to WebSocket");
            document.getElementById('connection-status').classList.remove('disconnected');
            document.getElementById('connection-status').classList.add('connected');
            document.getElementById('connection-status').innerHTML = '<i class="fas fa-plug"></i> Connected';
            document.getElementById('loading-spinner').style.display = "none";
        };

        ws.onclose = function() {
            console.log("Disconnected from WebSocket");
            document.getElementById('connection-status').classList.remove('connected');
            document.getElementById('connection-status').classList.add('disconnected');
            document.getElementById('connection-status').innerHTML = '<i class="fas fa-plug"></i> Disconnected';
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data).data;
            const tableBody = document.getElementById("data-table").getElementsByTagName("tbody")[0];

            // Clear current table
            tableBody.innerHTML = "";

            // Add each row of data to the table
            data.forEach(row => {
                const tr = document.createElement("tr");

                const timestampTd = document.createElement("td");
                timestampTd.textContent = row["Timestamp"];
                tr.appendChild(timestampTd);

                const classTd = document.createElement("td");
                classTd.textContent = row["Class"];
                tr.appendChild(classTd);

                const confidenceTd = document.createElement("td");
                confidenceTd.textContent = row["Confidence"] + "%";
                tr.appendChild(confidenceTd);

                const violationTd = document.createElement("td");
                violationTd.textContent = row["Restricted Area Violation"];
                tr.appendChild(violationTd);

                tableBody.appendChild(tr);
            });
        };