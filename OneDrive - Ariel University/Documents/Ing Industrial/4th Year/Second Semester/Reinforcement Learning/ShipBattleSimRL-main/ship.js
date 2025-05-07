// ship.js

class Ship {
    constructor(x, y, team, id, p_instance) {
        this.p = p_instance; // Store the p5 sketch instance reference
        this.id = id;
        this.team = team;
        this.position = this.p.createVector(x, y); // Use instance 'p' for instance methods

        this.velocity = this.p.createVector(0, 0);
        this.acceleration = this.p.createVector(0, 0); // Use instance 'p'
		
		// TODO: would be cool to play with these values
        this.maxSpeed = 2;
        this.maxForce = 0.15;
        this.size = 11;
        this.visionRadius = 100;     
        this.firingRange = 75;      
        this.life = 3;
        this.ammo = 10;             
        this.fireCooldown = 0;
        this.maxFireCooldown = 30;  
        this.fireChance = 0.08;  // TODO: remove, just for the demo, the DRL should make this decision

        this.collisionDamageCooldown = 0;
        this.maxCollisionDamageCooldown = 30; // 0.5 seconds cooldown for collision hits
		
		// TODO: remove, just for the demo, the DRL should make this decision
        // Wander parameters (same)
        this.wanderTheta = this.p.random(this.p.TWO_PI);
        this.wanderRadius = 10;
        this.wanderDistance = 20;
        this.wanderChange = 0.3;

        this.hullColor = (team === 'blue') ? this.p.color(0, 80, 200) : this.p.color(200, 40, 40);
        this.cabinColor = (team === 'blue') ? this.p.color(100, 150, 255) : this.p.color(255, 100, 100);
        this.detailColor = this.p.color(100); // Grey details
        this.strokeColor = this.p.color(40);

        this.currentTarget = null; // Store the ship being targeted
        this.lastShotFired = 0; // Track when last shot was fired (for visualization)
		
        this.timeOfDeath = null;        // Timestamp when life reached 0
        this.markedForRemoval = false;  // Flag to signal removal in the next game loop iteration
    }

    isDead() {
        return this.life <= 0;
    }

    applyForce(force) {
        this.acceleration.add(force);
    }

    // --- Steering Behaviors ---
    wander() {
        let circlePos = this.velocity.copy().normalize().mult(this.wanderDistance).add(this.position);
        this.wanderTheta += this.p.random(-this.wanderChange, this.wanderChange);
        let displacement = this.p.createVector(this.wanderRadius * this.p.cos(this.wanderTheta), this.wanderRadius * this.p.sin(this.wanderTheta));
        // Use global p5 for static vector math is generally safer in instance mode
        let wanderForce = p5.Vector.sub(circlePos.add(displacement), this.position);
        wanderForce.setMag(this.maxForce);
        return wanderForce;
    }

    avoid(obstacles, mapW, mapH, baseW) {
        let avoidanceForce = this.p.createVector(0, 0);
        let predictionDistance = this.velocity.mag() * 25;
        let futurePos = p5.Vector.add(this.position, this.velocity.copy().setMag(predictionDistance));
        let shipRadius = this.size;
        let margin = shipRadius * 1.5;
        let steerStrength = this.maxForce * 2.5;
        let minX = (this.team === 'blue') ? -Infinity : baseW;
        let maxX = (this.team === 'red') ? Infinity : mapW - baseW;

        if (futurePos.x < minX + margin) avoidanceForce.add(this.p.createVector(steerStrength, 0));
        if (futurePos.x > maxX - margin) avoidanceForce.add(this.p.createVector(-steerStrength, 0));
        if (futurePos.y < margin) avoidanceForce.add(this.p.createVector(0, steerStrength));
        if (futurePos.y > mapH - margin) avoidanceForce.add(this.p.createVector(0, -steerStrength));

        obstacles.forEach(island => {
            let distToIslandCenter = p5.Vector.dist(futurePos, island.position);
            let combinedRadii = island.boundingRadius + shipRadius * 1.8;
            if (distToIslandCenter < combinedRadii) {
                let awayFromIsland = p5.Vector.sub(futurePos, island.position);
                let strength = this.maxForce * 2.0 * this.p.map(distToIslandCenter, 0, combinedRadii, 1.5, 0.5);
                awayFromIsland.setMag(strength);
                avoidanceForce.add(awayFromIsland);
            }
        });
        return avoidanceForce;
    }

    separate(allShips) {
       let desiredSeparation = this.size * 3.5;
       let separationForce = this.p.createVector(0, 0);
       let count = 0;
       allShips.forEach(other => {
           if (other !== this && !other.isDead()) {
               let d = p5.Vector.dist(this.position, other.position);
               if (d > 0 && d < desiredSeparation) {
                   let diff = p5.Vector.sub(this.position, other.position);
                   diff.normalize();
                   diff.div(d);
                   separationForce.add(diff);
                   count++;
               }
           }
       });
       if (count > 0) {
           separationForce.div(count);
           separationForce.setMag(this.maxForce * 1.6);
       }
       return separationForce;
    }

    applySteering(allShips, islands, mapW, mapH, baseW) {
       if (this.isDead()) return;
       let wanderForce = this.wander();
       let avoidanceForce = this.avoid(islands, mapW, mapH, baseW);
       let separationForce = this.separate(allShips);

       wanderForce.mult(0.5);
       avoidanceForce.mult(1.8);
       separationForce.mult(1.6);

       this.applyForce(wanderForce);
       this.applyForce(avoidanceForce);
       this.applyForce(separationForce);
    }

    takeHit(damageSource = 'projectile') {
        if (this.isDead()) return; // Already dead, do nothing

        // Apply damage cooldown logic for collisions
        if (damageSource === 'collision') {
            if (this.collisionDamageCooldown > 0) return; // On collision cooldown
            this.collisionDamageCooldown = this.maxCollisionDamageCooldown;
        }

        this.life--; // Decrease life

        // Check if the ship just died
        if (this.life <= 0) {
            this.life = 0; // Ensure life doesn't go below 0
            if (this.timeOfDeath === null) { // Record time only once
                this.timeOfDeath = this.p.millis();
                // Optional: Stop the ship immediately upon death
                this.velocity.mult(0);
                this.acceleration.mult(0);
                this.currentTarget = null; // Stop targeting
                console.log(`${this.id} destroyed at time ${this.timeOfDeath}`); // Debug
            }
        }
    }

    findAndEngageTarget(allShips) {
        if (this.isDead() || this.ammo <= 0 || this.fireCooldown > 0) {
            this.currentTarget = null; // Cannot engage
            return;
        }

        let potentialTargets = [];
        // Find living enemies within vision range
        allShips.forEach(ship => {
            if (ship.team !== this.team && !ship.isDead()) {
                let d = p5.Vector.dist(this.position, ship.position);
                if (d < this.visionRadius) {
                    potentialTargets.push({ ship: ship, distance: d });
                }
            }
        });

        if (potentialTargets.length === 0) {
            this.currentTarget = null;
            return; // No targets in vision
        }

        // Sort targets by distance (closest first)
        potentialTargets.sort((a, b) => a.distance - b.distance);

        // Filter for targets within firing range
        let targetsInRange = potentialTargets.filter(t => t.distance <= this.firingRange);

        if (targetsInRange.length > 0) {
            // Target the closest enemy in firing range
            this.currentTarget = targetsInRange[0].ship;
            let targetDistance = targetsInRange[0].distance;

            // Random chance to fire this frame
            if (this.p.random(1) < this.fireChance) {
                this.fire(this.currentTarget, targetDistance);
            }
        } else {
            // Targets are visible but out of range - clear current fire target
            this.currentTarget = null;
        }
    }

    fire(target, distance) {
        if (this.ammo <= 0 || this.fireCooldown > 0 || this.isDead() || target.isDead()) {
            return false;
        }

        // Calculate Hit Probability: 100% at 0 dist, 10% at 30 dist
        // Formula: HitChance = max(0, 1.0 - 0.03 * distance)
        let hitProb = Math.max(0, 1.0 - 0.03 * distance);
        hitProb = this.p.constrain(hitProb, 0, 1.0); // Ensure valid probability

        this.ammo--;
        this.fireCooldown = this.maxFireCooldown;
        this.lastShotFired = this.p.millis(); // Record time of shot for visual effect

        // Roll for hit
        if (this.p.random(1) < hitProb) {
            target.takeHit('projectile'); // Target takes projectile damage
            return true;
        } else {
            return false;
        }
    }

    update(islands, allShips, mapWidth, mapHeight, baseWidth) {
        if (this.isDead()) {
            this.velocity.mult(0.9);
            this.position.add(this.velocity);
            this.currentTarget = null; // Dead ships don't target
            return;
        }

        // Steering and Movement
        this.applySteering(allShips, islands, mapWidth, mapHeight, baseWidth);
        this.velocity.add(this.acceleration);
        this.velocity.limit(this.maxSpeed);
        let nextPos = p5.Vector.add(this.position, this.velocity);

        // Boundary checks and position update
        let boundaryHit = this.checkAndEnforceBoundaries(nextPos, mapWidth, mapHeight, baseWidth);
        if (boundaryHit) this.takeHit('collision'); // Apply damage if boundary hit
        this.position.set(nextPos); // Set final position for this frame
        this.acceleration.mult(0); // Reset acceleration

        // Cooldowns
        if (this.collisionDamageCooldown > 0) this.collisionDamageCooldown--;
        if (this.fireCooldown > 0) this.fireCooldown--; // Decrement fire cooldown


        // Combat Logic (Find and Engage) - Moved after position update and cooldown decrement
        this.findAndEngageTarget(allShips);


        // Island Collision Check (after final position is set)
        this.checkIslandCollisions(islands);

    }

    checkAndEnforceBoundaries(nextPos, mapWidth, mapHeight, baseWidth) {
        let boundaryHit = false;
        let effectiveRadius = this.size * 0.8;
        let minX = (this.team === 'blue') ? 0 : baseWidth;
        let maxX = (this.team === 'red') ? mapWidth : mapWidth - baseWidth;
        let minY = 0;
        let maxY = mapHeight;

        if (nextPos.x - effectiveRadius < minX) { nextPos.x = minX + effectiveRadius; this.velocity.x = 0; boundaryHit = true; }
        else if (nextPos.x + effectiveRadius > maxX) { nextPos.x = maxX - effectiveRadius; this.velocity.x = 0; boundaryHit = true; }
        if (nextPos.y - effectiveRadius < minY) { nextPos.y = minY + effectiveRadius; this.velocity.y = 0; boundaryHit = true; }
        else if (nextPos.y + effectiveRadius > maxY) { nextPos.y = maxY - effectiveRadius; this.velocity.y = 0; boundaryHit = true; }
        return boundaryHit;
    }

     checkIslandCollisions(islands) {
         islands.forEach(island => {
            // Use p5.Vector.dist for potentially slightly better accuracy/readability
            let d = p5.Vector.dist(this.position, island.position);
            let collisionThreshold = this.size * 0.8 + island.boundingRadius;
            if (d < collisionThreshold) {
                 this.takeHit('collision');
                 let overlap = collisionThreshold - d;
                 // Ensure pushDirection is calculated correctly
                 let pushDirection = p5.Vector.sub(this.position, island.position).normalize();
                 // Check if pushDirection is valid before applying mult
                 if (pushDirection.magSq() > 0) { // Avoid NaN if somehow position == island.position
                    this.position.add(pushDirection.mult(overlap * 1.1));
                    let normalVelocity = this.velocity.dot(pushDirection);
                    if(normalVelocity < 0) {
                        this.velocity.sub(pushDirection.mult(normalVelocity * 1.2));
                    }
                 }
            }
        });
    }

    display() {
        // Color setup and fade for dead ships (same as before)
        let currentHullColor = this.p.color(this.hullColor);
        let currentCabinColor = this.p.color(this.cabinColor);
        let currentDetailColor = this.p.color(this.detailColor);
        let currentStrokeColor = this.p.color(this.strokeColor);
        if (this.isDead()) {
            currentHullColor.setAlpha(80); currentCabinColor.setAlpha(80);
            currentDetailColor.setAlpha(80); currentStrokeColor.setAlpha(100);
        }

        this.p.push();
        this.p.translate(this.position.x, this.position.y);

        // --- Draw Debug Circles (if enabled and alive) ---
        if (!this.isDead()) {
            this.p.noFill();
            this.p.strokeWeight(3);
            // Use global flags set in script.js
            if (typeof showVision !== 'undefined' && showVision) {
                this.p.stroke(0, 255, 0, 50); // Green, semi-transparent
                this.p.ellipse(0, 0, this.visionRadius * 2);
            }
            if (typeof showFiringRange !== 'undefined' && showFiringRange) {
                this.p.stroke(255, 165, 0, 50); // Orange, semi-transparent
                this.p.ellipse(0, 0, this.firingRange * 2);
            }
        }

        // --- Draw Ship Body (Rotated) ---
        this.p.rotate(this.velocity.heading()); // Rotate after drawing circles

        this.p.strokeWeight(1.5);
        this.p.stroke(currentStrokeColor); // Use potentially faded stroke for dead ship

        // Draw ship components (hull, cabin, details - same as before)
        let s = this.size;
        this.p.fill(currentHullColor);   this.p.beginShape(); this.p.vertex(s * 1.4, 0); this.p.vertex(s * 0.2, -s * 0.7); this.p.vertex(-s * 1.1, -s * 0.6); this.p.vertex(-s * 0.9, 0); this.p.vertex(-s * 1.1, s * 0.6); this.p.vertex(s * 0.2, s * 0.7); this.p.endShape(this.p.CLOSE);
        this.p.fill(currentCabinColor); this.p.stroke(currentStrokeColor); this.p.rect(-s * 0.7, -s * 0.4, s * 1.1, s * 0.8, s * 0.1);
        this.p.stroke(currentDetailColor); this.p.strokeWeight(1); this.p.line(s * 0.3, -s * 0.3, s * 0.8, -s * 0.3); this.p.line(s * 0.3, s * 0.3, s * 0.8, s * 0.3);
        this.p.fill(currentDetailColor); this.p.noStroke(); this.p.ellipse(-s * 0.3, 0, s * 0.3, s * 0.3);

        // --- Health Bar ---
        if (!this.isDead()) {
            this.displayHealthBar(s);
        }

        // --- Muzzle Flash Visualization ---
         let timeSinceShot = this.p.millis() - this.lastShotFired;
         if (!this.isDead() && timeSinceShot < 150) { // Flash for 150ms
             let flashAlpha = this.p.map(timeSinceShot, 0, 150, 200, 0);
             this.p.fill(255, 255, 0, flashAlpha); // Yellow flash
             this.p.noStroke();
             this.p.ellipse(s * 1.4, 0, 8, 8); // Flash at the nose
         }

        this.p.pop(); // Restore drawing state (undo translate/rotate)

        // --- Draw Target Line (outside push/pop to use absolute coords) ---
        if (typeof showTargetLines !== 'undefined' && showTargetLines && !this.isDead() && this.currentTarget && !this.currentTarget.isDead()) {
             this.p.strokeWeight(1);
             this.p.stroke(255, 0, 0, 100); // Dim red line
             this.p.line(this.position.x, this.position.y, this.currentTarget.position.x, this.currentTarget.position.y);
        }
    }

    displayHealthBar(shipSize) {
        let s = shipSize;
        let healthRatio = this.life / 3;
        this.p.noStroke();
        let barX = -s * 0.9; let barY = s * 1.1; let barW = s * 1.8; let barH = 5;
        this.p.fill(80, 180); this.p.rect(barX, barY, barW, barH, 2);
        if (healthRatio > 0.6) this.p.fill(0, 200, 0, 230);
        else if (healthRatio > 0.3) this.p.fill(255, 200, 0, 230);
        else this.p.fill(200, 0, 0, 230);
        this.p.rect(barX, barY, barW * healthRatio, barH, 2);
    }

} 