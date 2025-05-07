// script.js - Main page logic and interaction
let showVision = false;
let showFiringRange = false;
let showTargetLines = true; // Default target lines to on

document.addEventListener('DOMContentLoaded', () => {
    const setupPage = document.getElementById('setup-page');
    const gamePage = document.getElementById('game-page');
    const trainingPage = document.getElementById('training-page');

    const setupForm = document.getElementById('setup-form');
    const btnShowSetup = document.getElementById('btn-show-setup');
    const btnShowTraining = document.getElementById('btn-show-training');
    const btnRestartSetup = document.getElementById('btn-restart-setup');
    const btnTrainingToSetup = document.getElementById('btn-training-to-setup');
	
    const checkShowVision = document.getElementById('check-show-vision');
    const checkShowFiring = document.getElementById('check-show-firing');
    const checkShowTargetLines = document.getElementById('check-show-target-lines');

	const gameEndModalElement = document.getElementById('gameEndModal');
	const gameEndModalLabel = document.getElementById('gameEndModalLabel');
	const resetButtonModal = document.getElementById('btn-reset-game-modal');
	const menuButtonModal = document.getElementById('btn-menu-modal');
	
	const gameEndModal = new bootstrap.Modal(gameEndModalElement); 

	
	resetButtonModal.addEventListener('click', () => {
	  gameEndModal.style.opacity = 0; // Hide the modal first
	  console.log("Reset Game button clicked!");
	  // TODO: develop this logic
	});

	menuButtonModal.addEventListener('click', () => {
	  gameEndModal.style.opacity = 0; // Hide the modal first
	  console.log("Main Menu button clicked!");
	  window.location.href = '/';
	});


    // --- Page Navigation ---

    function showPage(pageId) {
        // Use Bootstrap classes for hiding/showing if preferred,
        // but direct style manipulation is also fine here.
        setupPage.style.display = 'none';
        gamePage.style.display = 'none';
        trainingPage.style.display = 'none';

        const pageToShow = document.getElementById(pageId);
        if (pageToShow) {
            pageToShow.style.display = 'block';
        } else {
            console.error("Page ID not found:", pageId);
             setupPage.style.display = 'block'; // Fallback to setup
        }


        // Stop game visualization if navigating away from game page
        if (pageId !== 'game-page' && typeof stopGameVisualization === 'function') {
             stopGameVisualization();
        }
    }

    // Navbar button event listeners
     if(btnShowSetup) btnShowSetup.addEventListener('click', () => showPage('setup-page'));
     if(btnShowTraining) btnShowTraining.addEventListener('click', () => showPage('training-page'));
     // Other button listeners
    if(btnRestartSetup) btnRestartSetup.addEventListener('click', () => showPage('setup-page'));
    if(btnTrainingToSetup) btnTrainingToSetup.addEventListener('click', () => showPage('setup-page'));

    // --- Game Setup ---
    if (setupForm) {
        setupForm.addEventListener('submit', (event) => {
            event.preventDefault(); // Prevent default form submission

            const numBlueShips = parseInt(document.getElementById('num-blue-ships').value, 10);
            const numRedShips = parseInt(document.getElementById('num-red-ships').value, 10);
            const numIslands = parseInt(document.getElementById('num-islands').value, 10);

            if (isNaN(numBlueShips) || numBlueShips < 3 || numBlueShips > 9 ||
                isNaN(numRedShips) || numRedShips < 3 || numRedShips > 9 ||
                isNaN(numIslands) || numIslands < 0 || numIslands > 10) {
                alert('Please enter valid numbers for ships (3-9) and islands (0-10).');
                return;
            }

            const gameConfiguration = {
                numBlueShips,
                numRedShips,
                numIslands
            };

             if (typeof startGameVisualization === 'function') {
                 startGameVisualization(gameConfiguration);
                 showPage('game-page'); // Switch to the game page
             } else {
                 console.error("startGameVisualization function not found!");
                 alert("Error starting game visualization.");
             }
        });
    } else {
        console.error("Setup form not found");
    }
	
    // --- Visualization Checkbox Listeners ---
    if (checkShowVision) {
        checkShowVision.addEventListener('change', (event) => {
            showVision = event.target.checked;
        });
    }
     if (checkShowFiring) {
        checkShowFiring.addEventListener('change', (event) => {
            showFiringRange = event.target.checked;
        });
    }
     if (checkShowTargetLines) {
         // Set initial state based on default checked status
         showTargetLines = checkShowTargetLines.checked;
         checkShowTargetLines.addEventListener('change', (event) => {
            showTargetLines = event.target.checked;
        });
     }


    // --- Initial State ---
    showPage('setup-page'); // Show setup page by default

}); // End DOMContentLoaded
