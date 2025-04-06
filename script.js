document.addEventListener('DOMContentLoaded', function () {
    const toggleButton = document.getElementById('toggleButton');
    const body = document.body;
    const container = document.querySelector('.container');
    const heading = document.querySelector('h1');
    const headings = document.querySelectorAll('h1, h2');
    const fileInput = document.getElementById('file-input');
    const fileInputLabel = document.querySelector('.file-input-label');
    const resultText = document.getElementById('result');
    const clearButton = document.getElementById('clearButton');
    const confidenceText = document.getElementById('confidence');
    const progressBarContainer = document.getElementById('progress-bar-container');
    const progressBar = document.getElementById('progress-bar');
    const submitButton = document.getElementById('submitButton');
    const uploadForm = document.getElementById('upload-form');
    const errorMessage = document.getElementById('error-message');
    const loadingSpinner = document.getElementById('loading-spinner');
    const statsContainer = document.getElementById('stats-container');
    const predictionHistory = document.getElementById('prediction-history');
    const statItems = document.querySelectorAll('.stat-item');

    // Store last stats to prevent unnecessary updates that trigger Five Server reloads
    let lastStats = null;

    // Function to toggle light mode
    function toggleLightMode() {
        body.classList.toggle('light-mode');
        container.classList.toggle('light-mode');
        headings.forEach(heading => heading.classList.toggle('light-mode'));
        fileInput.classList.toggle('light-mode');
        fileInputLabel.classList.toggle('light-mode');
        resultText.classList.toggle('light-mode');
        clearButton.classList.toggle('light-mode');
        confidenceText.classList.toggle('light-mode');
        toggleButton.classList.toggle('light-mode');
        statsContainer.classList.toggle('light-mode');
        predictionHistory.classList.toggle('light-mode');
        statItems.forEach(item => item.classList.toggle('light-mode'));
        
        // Update the icon
        const icon = toggleButton.querySelector('i');
        if (body.classList.contains('light-mode')) {
            icon.className = 'fas fa-sun';
        } else {
            icon.className = 'fas fa-moon';
        }

        // Save the current mode preference
        const isLightMode = body.classList.contains('light-mode');
        localStorage.setItem('lightMode', isLightMode);
    }

    // Initialize mode based on saved preference
    const savedMode = localStorage.getItem('lightMode');
    if (savedMode === 'true' && !body.classList.contains('light-mode')) {
        toggleLightMode();
    } else if (savedMode === null && !body.classList.contains('light-mode')) {
        // Default to light mode on first visit
        toggleLightMode();
    }

    // Add event listener to the toggle button
    toggleButton.addEventListener('click', toggleLightMode);

    // File input styling
    fileInputLabel.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('file-input-dragover');
    });

    fileInputLabel.addEventListener('dragleave', function() {
        this.classList.remove('file-input-dragover');
    });

    fileInputLabel.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('file-input-dragover');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });

    // Image Preview
    function handleFileSelect() {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                const imagePreview = document.getElementById('image-preview');
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                clearButton.style.display = 'inline-block';
                
                // Update label to show filename
                const filename = file.name.length > 20 ? file.name.substring(0, 17) + '...' : file.name;
                fileInputLabel.querySelector('span').textContent = filename;
            };
            reader.readAsDataURL(file);
        }
    }

    fileInput.addEventListener('change', handleFileSelect);

    // Clear Button functionality
    clearButton.addEventListener('click', function () {
        fileInput.value = '';
        document.getElementById('image-preview').src = '';
        document.getElementById('image-preview').style.display = 'none';
        resultText.textContent = '';
        confidenceText.textContent = '';
        this.style.display = 'none';
        progressBarContainer.style.display = 'none';
        progressBar.style.width = '0%';
        errorMessage.textContent = '';
        fileInputLabel.querySelector('span').textContent = 'Choose an image or drag it here';
    });

    // Prevent default form submission
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
    });

    // Handle classification
    submitButton.addEventListener('click', function(event) {
        event.preventDefault();
        
        if (fileInput.files.length === 0) {
            errorMessage.textContent = 'Please select a file!';
            errorMessage.className = 'error';
            return;
        }

        errorMessage.textContent = '';
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Show progress bar and loading spinner
        progressBarContainer.style.display = 'block';
        progressBar.style.width = '0%';
        loadingSpinner.style.display = 'block';
        submitButton.disabled = true;

        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress > 90) clearInterval(progressInterval);
            progressBar.style.width = `${progress}%`;
        }, 100);

        // Use fetch API instead of XMLHttpRequest
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            clearInterval(progressInterval);
            progressBar.style.width = '100%';
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.prediction) {
                resultText.textContent = `Prediction: ${data.prediction}`;
                resultText.className = 'prediction';
                confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                
                // Update UI with prediction result
                document.getElementById('image-preview').style.display = 'block';
                clearButton.style.display = 'inline-flex';
                
                // Update stats display without reloading page
                updateUIWithPrediction(data);
                
                // We'll update stats after a short delay to ensure the prediction is saved
                setTimeout(updateStats, 500);
            } else {
                resultText.textContent = `Error: ${data.error || 'Unexpected response'}`;
                confidenceText.textContent = '';
                resultText.className = 'error';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.className = 'error';
        })
        .finally(() => {
            // Hide progress bar and loading spinner after a short delay
            setTimeout(() => {
                progressBarContainer.style.display = 'none';
                progressBar.style.width = '0%';
                loadingSpinner.style.display = 'none';
                submitButton.disabled = false;
            }, 500);
        });
    });

    // Update UI with new prediction without waiting for server response
    function updateUIWithPrediction(prediction) {
        const historyList = document.getElementById('history-list');
        if (!historyList) return;
        
        const listItem = document.createElement('li');
        listItem.className = 'prediction-item';
        if (body.classList.contains('light-mode')) {
            listItem.classList.add('light-mode');
        }
        
        // Format the timestamp
        const timestamp = new Date(prediction.timestamp).toLocaleString();
        
        // Get bird icon based on prediction
        let birdIcon = 'fa-dove';
        if (prediction.prediction.includes('robin')) {
            birdIcon = 'fa-kiwi-bird';
        } else if (prediction.prediction.includes('unknown')) {
            birdIcon = 'fa-question-circle';
        }
        
        // Format the prediction item
        listItem.innerHTML = `
            <div class="prediction-class"><i class="fas ${birdIcon}"></i>${prediction.prediction}</div>
            <div class="prediction-confidence">${(prediction.confidence * 100).toFixed(2)}%</div>
            <div class="prediction-time ${body.classList.contains('light-mode') ? 'light-mode' : ''}">${timestamp}</div>
        `;
        
        // Insert at the beginning of the list
        historyList.insertBefore(listItem, historyList.firstChild);
        
        // Add fancy entrance animation
        listItem.style.opacity = '0';
        listItem.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            listItem.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            listItem.style.opacity = '1';
            listItem.style.transform = 'translateY(0)';
        }, 10);
        
        // Limit to last 10 predictions
        while (historyList.children.length > 10) {
            historyList.removeChild(historyList.lastChild);
        }
        
        // Show history container
        document.getElementById('prediction-history').style.display = 'block';
        document.getElementById('stats-container').style.display = 'block';
    }

    // Stats update function
    async function updateStats() {
        try {
            const response = await fetch('http://127.0.0.1:5000/stats');
            const stats = await response.json();
            
            // Skip updates if stats haven't changed to prevent triggering Five Server reloads
            if (lastStats && 
                lastStats.total_predictions === stats.total_predictions &&
                lastStats.average_confidence === stats.average_confidence) {
                return;
            }
            
            lastStats = stats;
            
            if (!stats || stats.error) {
                console.error('Stats error:', stats?.error || 'No stats available');
                return;
            }
            
            // Update stats display with animation
            animateCounter('total-predictions', stats.total_predictions);
            animateCounter('avg-confidence', (stats.average_confidence * 100).toFixed(2) + '%');
            animateCounter('avg-processing-time', (stats.average_processing_time * 1000).toFixed(2) + 'ms');

            // Update prediction history only if it's different
            if (stats.history && stats.history.length > 0) {
                updateHistoryDisplay(stats.history);
            }

            // Make sure containers are visible
            document.getElementById('stats-container').style.display = 'block';
            document.getElementById('prediction-history').style.display = 'block';
            
        } catch (error) {
            console.error('Error updating stats:', error);
        }
    }

    // Animate counter function
    function animateCounter(elementId, targetValue) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const startValue = element.textContent;
        const start = parseFloat(startValue.replace(/[^0-9.-]+/g, '')) || 0;
        const target = parseFloat(targetValue.toString().replace(/[^0-9.-]+/g, ''));
        const suffix = targetValue.toString().replace(/[0-9.-]+/g, '');
        
        const duration = 1000;
        const startTime = performance.now();
        
        function updateCounter(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const value = start + (target - start) * progress;
            element.textContent = value.toFixed(1).replace(/\.0$/, '') + suffix;
            
            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            } else {
                element.textContent = targetValue;
            }
        }
        
        requestAnimationFrame(updateCounter);
    }

    function updateHistoryDisplay(history) {
        const historyList = document.getElementById('history-list');
        if (!historyList) return;
        
        // Clear existing history only if we need to update
        historyList.innerHTML = '';
        
        history.forEach(prediction => {
            const listItem = document.createElement('li');
            listItem.className = 'prediction-item';
            if (body.classList.contains('light-mode')) {
                listItem.classList.add('light-mode');
            }
            
            const timestamp = new Date(prediction.timestamp).toLocaleString();
            
            // Get bird icon based on prediction
            let birdIcon = 'fa-dove';
            if (prediction.prediction.includes('robin')) {
                birdIcon = 'fa-kiwi-bird';
            } else if (prediction.prediction.includes('unknown')) {
                birdIcon = 'fa-question-circle';
            }
            
            listItem.innerHTML = `
                <div class="prediction-class"><i class="fas ${birdIcon}"></i>${prediction.prediction}</div>
                <div class="prediction-confidence">${(prediction.confidence * 100).toFixed(2)}%</div>
                <div class="prediction-time ${body.classList.contains('light-mode') ? 'light-mode' : ''}">${timestamp}</div>
            `;
            historyList.appendChild(listItem);
        });
    }

    // Initial stats load - once only
    updateStats();
    
    // Update stats less frequently to reduce file change triggers
    setInterval(updateStats, 60000);  // Update every 60 seconds
});