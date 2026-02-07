document.addEventListener('DOMContentLoaded', () => {
    // CloudPeak Astro: Scroll-triggered animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    // Observe all fade-in elements
    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

    // Animated counter for stats
    const animateCounter = (element, target, duration = 2000) => {
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current);
            }
        }, 16);
    };

    // Trigger counter animation when stats are visible
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.dataset.animated) {
                entry.target.dataset.animated = 'true';
                const counters = entry.target.querySelectorAll('.counter');
                counters.forEach(counter => {
                    const text = counter.textContent;
                    const number = parseFloat(text);
                    if (!isNaN(number)) {
                        counter.textContent = '0';
                        animateCounter(counter, number);
                    }
                });
            }
        });
    }, { threshold: 0.5 });

    document.querySelectorAll('.hero-stats').forEach(el => statsObserver.observe(el));

    const uploadForm = document.getElementById('upload-form');
    const predictionResult = document.getElementById('prediction-result');
    const uploadedImage = document.getElementById('uploaded-image');
    const imageInput = document.getElementById('image');
    const fileNameDisplay = document.getElementById('file-name-display');
    const resultContainer = document.getElementById('result-container');
    const fileUploadPrompt = document.querySelector('.file-upload-wrapper p');

    if (imageInput && fileNameDisplay && fileUploadPrompt) {
        imageInput.addEventListener('change', () => {
            if (imageInput.files.length > 0) {
                fileNameDisplay.textContent = imageInput.files[0].name;
                fileUploadPrompt.style.display = 'none';
            } else {
                fileNameDisplay.textContent = 'No file chosen';
                fileUploadPrompt.style.display = 'block';
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(uploadForm);
            const imageFile = formData.get('image');

            if (!imageFile || imageFile.size === 0) {
                predictionResult.textContent = 'Please select an image file.';
                resultContainer.classList.add('visible');
                return;
            }

            predictionResult.textContent = 'Analyzing...';
            uploadedImage.style.display = 'none';
            resultContainer.classList.remove('visible');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (response.ok) {
                    predictionResult.textContent = data.prediction;
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        uploadedImage.src = event.target.result;
                        uploadedImage.style.display = 'block';
                    }
                    reader.readAsDataURL(imageFile);
                } else {
                    predictionResult.textContent = `Error: ${data.error}`;
                }
            } catch (error) {
                predictionResult.textContent = 'An unexpected error occurred.';
                console.error('Prediction error:', error);
            } finally {
                resultContainer.classList.add('visible');
                // Hide analyze button when results show
                const analyzeBtn = document.getElementById('analyze-btn');
                if (analyzeBtn) {
                    analyzeBtn.classList.add('hidden');
                }
                // Add has-results class to grid for 2-column layout
                const analysisGrid = document.querySelector('.analysis-grid');
                if (analysisGrid) {
                    analysisGrid.classList.add('has-results');
                }
                try {
                    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } catch (e) {
                    // no-op
                }
            }
        });
    }
});
