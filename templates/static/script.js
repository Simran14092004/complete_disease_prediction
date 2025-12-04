// JWT Token Handling
function getAuthToken() {
    return localStorage.getItem('access_token');
}

async function fetchWithAuth(url, options = {}) {
    const token = getAuthToken();
    const headers = {
        ...options.headers,
        'Authorization': `Bearer ${token}`
    };
    return fetch(url, { ...options, headers });
}

function checkAuthAndRenderUI() {
    const token = getAuthToken();
    if (token) {
        console.log('User is authenticated. Token found.');
        // Here you would typically decode the token to get user roles
        // and then show/hide UI elements accordingly.
        // For example, hide login/signup, show logout, show admin links.
        // let decodedToken = parseJwt(token); // You would need a parseJwt function
        // if (decodedToken.role === 'admin') { /* show admin elements */ }
    } else {
        console.log('User is not authenticated. No token found.');
        // Show login/signup, hide logout, hide protected content
    }
}

function logout() {
    localStorage.removeItem('access_token');
    alert('You have been logged out.');
    window.location.href = '/login'; // Redirect to login page
}

// Ensure UI is rendered based on auth status on page load
document.addEventListener('DOMContentLoaded', checkAuthAndRenderUI);

// Toggle Menu Visibility
const menuToggle = document.getElementById('menuToggle');
const menuNav = document.getElementById('menuNav');
const body = document.body;

menuToggle.addEventListener('click', (e) => {
    // Stop the click from propagating to the body
    e.stopPropagation();

    // Toggle visibility of the menu
    menuNav.classList.toggle('visible');

    // Add/remove active class for the hamburger menu animation
    menuToggle.classList.toggle('active');

    // Add/remove the body overlay when menu is open
    if (menuNav.classList.contains('visible')) {
        body.classList.add('menu-active');
    } else {
        body.classList.remove('menu-active');
    }
});

// Hide menu when clicking anywhere on the page
document.addEventListener('click', () => {
    if (menuNav.classList.contains('visible')) {
        menuNav.classList.remove('visible');
        menuToggle.classList.remove('active');
        body.classList.remove('menu-active');
    }
});

// Prevent menu from hiding when clicking inside the menu
menuNav.addEventListener('click', (e) => {
    e.stopPropagation();
});

// Smooth scrolling for navigation links
document.querySelectorAll('nav a').forEach((anchor) => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetSection = document.querySelector(this.getAttribute('href'));
        if (targetSection) {
            targetSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start',
            });
        }
    });
});

// Initialize sliders for sections
initializeSlider('.disease-slider');
initializeSlider('.article-slider');
initializeSlider('.course-slider');

/**
 * Initializes a slider for the given selector. Supports autoplay, drag, touch gestures, navigation buttons, and pagination.
 * @param {string} sliderSelector - The CSS selector for the slider container.
 */
function initializeSlider(sliderSelector) {
    const sliderContainer = document.querySelector(sliderSelector);
    if (!sliderContainer) return;

    const sliderWrapper = sliderContainer.querySelector('.slider-wrapper');
    const slides = Array.from(sliderWrapper.children);
    const paginationContainer = sliderContainer.querySelector('.slider-pagination');
    let slideIndex = 0;
    // Calculate slide width dynamically, including margins
    const slideWidth = slides[0].offsetWidth +
        (parseFloat(window.getComputedStyle(slides[0]).marginRight) || 0) +
        (parseFloat(window.getComputedStyle(slides[0]).marginLeft) || 0);

    let isDragging = false;
    let startPosition = 0;
    let currentPosition = 0;

    // Create pagination dots
    function createPaginationDots() {
        if (!paginationContainer) return;
        paginationContainer.innerHTML = ''; // Clear existing dots
        for (let i = 0; i < slides.length; i++) {
            const dot = document.createElement('span');
            dot.classList.add('pagination-dot');
            if (i === slideIndex) {
                dot.classList.add('active');
            }
            dot.addEventListener('click', () => {
                slideIndex = i;
                updateSliderPosition();
                updatePaginationDots();
            });
            paginationContainer.appendChild(dot);
        }
    }

    // Update active pagination dot
    function updatePaginationDots() {
        if (!paginationContainer) return;
        const dots = paginationContainer.querySelectorAll('.pagination-dot');
        dots.forEach((dot, index) => {
            if (index === slideIndex) {
                dot.classList.add('active');
            } else {
                dot.classList.remove('active');
            }
        });
    }

    function updateSliderPosition() {
        sliderWrapper.style.transform = `translateX(${-slideIndex * slideWidth}px)`;
    }

    function nextSlide() {
        slideIndex = (slideIndex + 1) % slides.length;
        updateSliderPosition();
        updatePaginationDots(); // Update dots after slide
    }

    function prevSlide() {
        slideIndex = (slideIndex - 1 + slides.length) % slides.length;
        updateSliderPosition();
        updatePaginationDots(); // Update dots after slide
    }

    function startDrag(event) {
        isDragging = true;
        startPosition = getClientX(event) - currentPosition;
        sliderWrapper.style.transition = '';
    }

    function drag(event) {
        if (!isDragging) return;
        const x = getClientX(event);
        currentPosition = x - startPosition;
        sliderWrapper.style.transform = `translateX(${currentPosition}px)`;
    }

    function endDrag() {
        if (!isDragging) return;
        isDragging = false;

        const movedBy = currentPosition - (-slideIndex * slideWidth);
        if (movedBy > slideWidth / 3) {
            slideIndex = Math.max(slideIndex - 1, 0);
        } else if (movedBy < -slideWidth / 3) {
            slideIndex = Math.min(slideIndex + 1, slides.length - 1);
        }
        sliderWrapper.style.transition = 'transform 0.3s ease-out';
        updateSliderPosition();
        updatePaginationDots(); // Update dots after drag
        currentPosition = -slideIndex * slideWidth;
    }

    function getClientX(event) {
        return event.type.startsWith('touch') ? event.touches[0].clientX : event.clientX;
    }

    // Autoplay
    let autoPlayInterval = setInterval(nextSlide, 5000);

    // Event listeners for dragging
    sliderWrapper.addEventListener('mousedown', startDrag);
    sliderWrapper.addEventListener('touchstart', startDrag, { passive: true });
    sliderWrapper.addEventListener('mousemove', drag);
    sliderWrapper.addEventListener('touchmove', drag, { passive: true });
    sliderWrapper.addEventListener('mouseup', endDrag);
    sliderWrapper.addEventListener('mouseleave', endDrag);
    sliderWrapper.addEventListener('touchend', endDrag);

    // Pause autoplay on hover
    sliderWrapper.addEventListener('mouseenter', () => {
        clearInterval(autoPlayInterval);
    });

    sliderWrapper.addEventListener('mouseleave', () => {
        autoPlayInterval = setInterval(nextSlide, 5000);
    });

    // Navigation Buttons
    const nextButton = sliderContainer.querySelector('.slider-next');
    const prevButton = sliderContainer.querySelector('.slider-prev');
    if (nextButton) nextButton.addEventListener('click', nextSlide);
    if (prevButton) prevButton.addEventListener('click', prevSlide);

    // Initial setup
    createPaginationDots();
    updateSliderPosition();
}

// Function to initialize charts dynamically based on user input
function initializeChart(chartId, chartType, labels, data) {
    const ctx = document.getElementById(chartId).getContext('2d');
    new Chart(ctx, {
        type: chartType,
        data: {
            labels: labels,
            datasets: [{
                label: 'User Input Data',
                data: data,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Event to handle user input and generate charts for different diseases
document.getElementById('generateCharts').addEventListener('click', () => {
    // Collect user inputs for Diabetes
    const glucose = parseFloat(document.getElementById('glucose').value) || 0;
    const bloodPressure = parseFloat(document.getElementById('bloodPressure').value) || 0;
    const age = parseFloat(document.getElementById('age').value) || 0;
    const skin_thickness = parseFloat(document.getElementById('skin_thickness').value) || 0;
    const insulin = parseFloat(document.getElementById('insulin').value) || 0;
    const pregnancies = parseFloat(document.getElementById('pregnancies').value) || 0;
    const diabetesInputs = [glucose, bloodPressure, insulin, ageDiabetes, skin_thickness, pregnancies];

    // Collect user inputs for Heart Disease
    const cholesterol = parseFloat(document.getElementById('cholesterol').value) || 0;
    const heartRate = parseFloat(document.getElementById('heartRate').value) || 0;
    const oldpeak = parseFloat(document.getElementById('oldpeak').value) || 0;
    const ageHeart = parseFloat(document.getElementById('ageHeart').value) || 0;
    const heartInputs = [cholesterol, heartRate, oldpeak, ageHeart];

    // Collect user inputs for Parkinson's
    const jitter_percent = parseFloat(document.getElementById('jitter_percent').value) || 0;
    const shimmer = parseFloat(document.getElementById('shimmer').value) || 0;
    const rpda = parseFloat(document.getElementById('rpda').value) || 0;
    const fo = parseFloat(document.getElementById('fo').value) || 0;
    const fhi = parseFloat(document.getElementById('fhi').value) || 0;
    const jitter_abs = parseFloat(document.getElementById('jitter_abs').value) || 0;
    const rap = parseFloat(document.getElementById('rapr').value) || 0;
    const shimmer_db = parseFloat(document.getElementById('shimmer_db').value) || 0;
    const apq = parseFloat(document.getElementById('apq').value) || 0;
    const apq3 = parseFloat(document.getElementById('apq3').value) || 0;
    const dda = parseFloat(document.getElementById('dda').value) || 0;
    const nhr = parseFloat(document.getElementById('nhr').value) || 0;
    const rpde = parseFloat(document.getElementById('rpde').value) || 0;
    const hnr = parseFloat(document.getElementById('hnr').value) || 0;
    const parkinsonInputs = [ fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer,
        shimmer_db, apq3, apq5, apq, dda, nhr, hnr ,rpda];
   

    // Collect user inputs for Depression
    const academicPressure = parseFloat(document.getElementById('academicPressure').value) || 0;
    const sleepDuration = parseFloat(document.getElementById('sleepDuration').value) || 0;
    const dietaryHabits = parseFloat(document.getElementById('dietaryHabits').value) || 0;
    const depressionInputs = [academicPressure, sleepDuration, dietaryHabits];

    // Collect user inputs for Generic Disease
    const symptom1 = parseFloat(document.getElementById('symptom1').value) || 0;
    const symptom2 = parseFloat(document.getElementById('symptom2').value) || 0;
    const symptom3 = parseFloat(document.getElementById('symptom3').value) || 0;
    const symptom4 = parseFloat(document.getElementById('symptom4').value) || 0;
    const genericInputs = [symptom1, symptom2, symptom3, symptom4];

    // Generate charts for each disease
    initializeChart('diabetesChart', 'bar', ['Glucose', 'Blood Pressure', 'BMI', 'Age'], diabetesInputs);
    initializeChart('heartChart', 'line', ['Cholesterol', 'Heart Rate', 'Oldpeak', 'Age'], heartInputs);
    initializeChart('parkinsonChart', 'pie', ['Jitter', 'Shimmer', 'HNR'], parkinsonInputs);
    initializeChart('depressionChart', 'radar', ['Academic Pressure', 'Sleep Duration', 'Dietary Habits'], depressionInputs);
    initializeChart('genericChart', 'bar', ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4'], genericInputs);
});




// Function to check if an element is in the viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Add 'visible' class to animated sections when they appear in viewport
function handleScrollAnimations() {
    const animatedSections = document.querySelectorAll('.animated-section');
    animatedSections.forEach(section => {
        if (isInViewport(section)) {
            section.classList.add('visible');
        }
    });
}

// Attach scroll event listener to trigger animations
window.addEventListener('scroll', handleScrollAnimations);

// Trigger animations on page load for elements already in viewport
document.addEventListener('DOMContentLoaded', handleScrollAnimations);



// Use GSAP for advanced animations
gsap.registerPlugin(ScrollTrigger);

// Fade-in animation for sections
gsap.from(".animated-section", {
    opacity: 0,
    duration: 1,
    y: 50,
    scrollTrigger: {
        trigger: ".animated-section",
        start: "top 80%", // Start animation when the section is 80% in the viewport
    }
});

// Slide-in animation for charts
gsap.from(".slide-in", {
    x: -200,
    opacity: 0,
    duration: 1,
    stagger: 0.2, // Delay between animations for multiple elements
    scrollTrigger: {
        trigger: ".slide-in",
        start: "top 80%",
    }
});