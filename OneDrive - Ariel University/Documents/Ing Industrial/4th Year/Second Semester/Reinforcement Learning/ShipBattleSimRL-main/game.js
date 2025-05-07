// game.js - p5.js sketch logic

let p5Instance = null; // Global reference to the p5 instance

// Game state variables
let shipsBlue = [];
let shipsRed = [];
let allShips = []; // Combined array for easier processing
let islands = [];
let gameConfig = {};
let mapWidth = 1000;
let mapHeight = 600;
let baseWidth = 50;
let gameOver = false;
let winner = null;

const REMOVAL_DELAY = 3000; // 3 seconds in milliseconds

// --- p5 Sketch Definition ---
const sketch = (p) => {

    p.setup = () => {
        const canvasContainer = document.getElementById('game-canvas-container');
        const canvas = p.createCanvas(mapWidth, mapHeight);
        canvas.parent(canvasContainer);
        p.frameRate(60);
        p.angleMode(p.RADIANS);
        console.log("p5 sketch setup complete.");
        initializeGameState(p);
    };

    p.draw = () => {
        p.background(112, 174, 225); // Updated water color #70aee1

        if (gameOver) {
            displayGameOver(p);
            // Keep drawing faded elements behind game over screen
             islands.forEach(island => island.display());
             allShips.forEach(ship => { // Draw remaining ships (some might be faded dead ones)
                if(!ship.markedForRemoval) ship.display();
            });
            return;
        }

        // --- Update and Display Islands ---
        islands.forEach(island => island.display());

        // --- Update and Display Ships ---
        allShips = [...shipsBlue, ...shipsRed]; // Update combined list

        // Collision check between ships
        handleShipCollisions(allShips);

        // Update ships (AI, physics, boundary checks, island collision)
        allShips.forEach(ship => {
             // Update using steering behaviors + internal collision checks
             ship.update(islands, allShips, mapWidth, mapHeight, baseWidth);
             // Display ships that are not yet marked for removal
             // The display method handles fading for dead ships during the 3s window
            if (!ship.markedForRemoval) {
                 ship.display();
            }
        });
		
        let currentTime = p.millis();
        allShips.forEach(ship => {
            // Check ships that are dead, have a death time recorded, and aren't already marked
            if (ship.isDead() && ship.timeOfDeath !== null && !ship.markedForRemoval) {
                if (currentTime - ship.timeOfDeath > REMOVAL_DELAY) {
                    ship.markedForRemoval = true;
                     // console.log(`Marking ${ship.id} for removal.`); // Debug
                }
            }
        });

        // --- Filter Arrays to Remove Marked Ships ---
        let shipsWereRemoved = false;
        const originalShipCount = allShips.length;

        shipsBlue = shipsBlue.filter(ship => !ship.markedForRemoval);
        shipsRed = shipsRed.filter(ship => !ship.markedForRemoval);

        // Rebuild the allShips array from the filtered team arrays
        allShips = [...shipsBlue, ...shipsRed];

        if (allShips.length < originalShipCount) {
            shipsWereRemoved = true;
            // console.log("Removed dead ships from arrays."); // Debug
        }

        // --- Check Game Over Condition ---
        checkGameOver();
    };

    // --- Helper Functions within the sketch ---

    const initializeGameState = (p_instance) => {
        gameOver = false;
        winner = null;
        shipsBlue = [];
        shipsRed = [];
        allShips = [];
        islands = [];

        // Create Islands (using new polygon generation)
        let islandAttempts = 0;
        const maxIslandAttempts = 50;
        const minIslandDist = 20; // Min distance between island bounding circles

        for (let i = 0; i < gameConfig.numIslands && islandAttempts < maxIslandAttempts; /* i++ below */) {
            islandAttempts++;
            // Estimate size between 10 and 100 (use as base radius estimate)
            let sizeEst = p_instance.random(20, 60); // Adjusted range for better results
            let buffer = sizeEst * 1.5; // Buffer from edges/bases based on max potential size

            let x = p_instance.random(baseWidth + buffer, mapWidth - baseWidth - buffer);
            let y = p_instance.random(buffer, mapHeight - buffer);
            let potentialIsland = new Island(x, y, sizeEst, p_instance);

            // Check overlap with existing islands using bounding radius
            let tooClose = false;
            for(let existingIsland of islands) {
                if (p5.Vector.dist(potentialIsland.position, existingIsland.position) < existingIsland.boundingRadius + potentialIsland.boundingRadius + minIslandDist) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                islands.push(potentialIsland);
                i++;
                 islandAttempts = 0; // Reset attempts counter
            }
        }
         if(islandAttempts >= maxIslandAttempts && gameConfig.numIslands > 0) {
             console.warn(`Could not place all ${gameConfig.numIslands} requested islands due to spacing constraints. Placed ${islands.length}.`);
         }


        // Create Blue Ships
        for (let i = 0; i < gameConfig.numBlueShips; i++) {
            let x = p_instance.random(15, baseWidth - 15);
            let y = p_instance.random(15, mapHeight - 15);
            shipsBlue.push(new Ship(x, y, 'blue', `B${i+1}`, p_instance));
        }

        // Create Red Ships
        for (let i = 0; i < gameConfig.numRedShips; i++) {
            let x = p_instance.random(mapWidth - baseWidth + 15, mapWidth - 15);
            let y = p_instance.random(15, mapHeight - 15);
            shipsRed.push(new Ship(x, y, 'red', `R${i+1}`, p_instance));
        }
        allShips = [...shipsBlue, ...shipsRed]; // Initial combined list

        console.log(`Game Initialized: ${shipsBlue.length} Blue, ${shipsRed.length} Red, ${islands.length} Islands.`);
    };

     // Function to handle ship-vs-ship collisions and apply damage
    const handleShipCollisions = (ships) => {
        for (let i = 0; i < ships.length; i++) {
            if (ships[i].isDead()) continue; // Skip dead ships

            for (let j = i + 1; j < ships.length; j++) {
                 if (ships[j].isDead()) continue; // Skip dead ships

                let shipA = ships[i];
                let shipB = ships[j];
                let minDist = shipA.size + shipB.size; // Collision distance based on size
                let d = p5.Vector.dist(shipA.position, shipB.position);

                if (d < minDist) {
                    // Collision detected! Apply damage to both
                    shipA.takeHit(true); // Pass true for collision damage (checks cooldown)
                    shipB.takeHit(true);

                     // Optional: Basic physics response - push apart slightly
                    let overlap = minDist - d;
                    let pushVector = p5.Vector.sub(shipA.position, shipB.position).normalize();
                    shipA.position.add(pushVector.copy().mult(overlap * 0.5));
                    shipB.position.add(pushVector.copy().mult(-overlap * 0.5));
                     // Dampen velocity towards each other? More complex physics.
                }
            }
        }
    };


    const checkGameOver = () => {
        // Check only if game isn't already marked as over
        if (gameOver) return;

        let blueAlive = shipsBlue.some(ship => !ship.isDead());
        let redAlive = shipsRed.some(ship => !ship.isDead());

        if (!blueAlive && redAlive) {
            gameOver = true;
            winner = 'Red';
            console.log("Game Over: Red Wins!");
            p.noLoop(); // Stop draw loop on game over
			showGameEndMessage("Red Team Wins!"); 
        } else if (!redAlive && blueAlive) {
            gameOver = true;
            winner = 'Blue';
             console.log("Game Over: Blue Wins!");
            p.noLoop(); // Stop draw loop
			showGameEndMessage("Blue Team Wins!"); 
        } else if (!redAlive && !blueAlive) {
            gameOver = true;
            winner = 'Draw'; // Or mutual destruction
            console.log("Game Over: Draw!");
            p.noLoop(); // Stop draw loop
			showGameEndMessage("Draw!"); 
        }
    };

    const displayGameOver = (p_instance) => {
        p_instance.push();
        // Dark semi-transparent overlay
        p_instance.fill(0, 0, 0, 190);
        p_instance.rect(0, 0, mapWidth, mapHeight);

        // Text styling
        p_instance.fill(255); // White text
        p_instance.textAlign(p_instance.CENTER, p_instance.CENTER);
        p_instance.textSize(52);
        p_instance.textFont('Arial'); // Or another common font
        p_instance.stroke(0); // Black outline for text
        p_instance.strokeWeight(2);

        let message = "";
        if (winner === 'Draw') {
            message = "Mutual Destruction!";
        } else {
            message = `${winner} Team Wins!`;
        }
        p_instance.text(message, mapWidth / 2, mapHeight / 2 - 20);

        p_instance.textSize(22);
        p_instance.strokeWeight(1);
        p_instance.fill(200); // Lighter grey for subtitle
        p_instance.text("Select 'New Game Setup' to play again.", mapWidth / 2, mapHeight / 2 + 40);
        p_instance.pop();
    };
    // Make reset function accessible globally
    window.resetGameSketch = () => {
        if (p) {
             p.loop(); // Ensure loop is running before re-init
             initializeGameState(p);
        }
    };

}; // End of sketch function definition

// --- Function to show the modal ---
function showGameEndMessage(message) {
  // Update the message text
  gameEndModalLabel.textContent = message; 
  // Show the modal
  gameEndModal.style.opacity = 1;
  gameEndModal.style.display  = "block";
}

// --- Function to create/destroy the p5 instance ---
function startGameVisualization(config) {
    if (p5Instance) {
        p5Instance.remove();
        p5Instance = null;
        console.log("Previous p5 instance removed.");
    }
    gameConfig = config;
    gameOver = false; // Reset game over flag
    winner = null;
    p5Instance = new p5(sketch);
    console.log("New p5 instance created.");
}

function stopGameVisualization() {
     if (p5Instance) {
        p5Instance.remove();
        p5Instance = null;
        shipsBlue = [];
        shipsRed = [];
        allShips = [];
        islands = [];
        document.getElementById('blue-stats').innerHTML = 'N/A';
        document.getElementById('red-stats').innerHTML = 'N/A';
        console.log("p5 instance stopped and removed.");
    }
}