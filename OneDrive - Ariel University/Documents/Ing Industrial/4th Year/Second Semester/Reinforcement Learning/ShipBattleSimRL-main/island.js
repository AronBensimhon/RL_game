// island.js

class Island {
    constructor(x, y, sizeEstimate, p5) {
        this.p = p5; // p5 instance reference
        this.position = this.p.createVector(x, y);
        this.color = this.p.color(139, 69, 19); // Brownish color (RGB)
        this.shoreColor = this.p.color(255, 223, 100); // Sandy yellow
        this.shoreThickness = 4;

        // Generate vertices for an irregular polygon
        this.vertices = [];
        let numVertices = this.p.floor(this.p.random(6, 12)); // 6 to 12 vertices
        let baseRadius = sizeEstimate * this.p.random(0.8, 1.2); // Overall size variation
        this.boundingRadius = 0; // Calculate actual max distance for collision

        for (let i = 0; i < numVertices; i++) {
            let angle = this.p.map(i, 0, numVertices, 0, this.p.TWO_PI);
            let r = baseRadius * this.p.random(0.6, 1.4);

            let vx = x + r * this.p.cos(angle);
            let vy = y + r * this.p.sin(angle);
            this.vertices.push(this.p.createVector(vx, vy));

            let distFromCenter = this.p.dist(x, y, vx, vy);
            if (distFromCenter > this.boundingRadius) {
                this.boundingRadius = distFromCenter;
            }
        }
         // Add a little padding to bounding radius
        this.boundingRadius += this.shoreThickness;
    }

    display() {
        this.p.push();

        // 1. Draw the shoreline (yellow stroke)
        this.p.strokeWeight(this.shoreThickness * 2); // Draw stroke wider
        this.p.stroke(this.shoreColor);
        this.p.fill(this.color); // Need fill otherwise stroke looks weird

        this.p.beginShape();
        this.vertices.forEach(v => {
            this.p.vertex(v.x, v.y);
        });
        this.p.endShape(this.p.CLOSE);

        // 2. Draw the main island body (brown fill, no stroke or thin dark)
        this.p.noStroke(); // Or thin dark stroke: this.p.stroke(50); this.p.strokeWeight(1);
        this.p.fill(this.color);

        this.p.beginShape();
        this.vertices.forEach(v => {
            this.p.vertex(v.x, v.y);
        });
        this.p.endShape(this.p.CLOSE);


        // --- Optional: Draw bounding circle for debugging ---
        /*
        this.p.noFill();
        this.p.stroke(255, 0, 0, 100);
        this.p.strokeWeight(1);
        this.p.ellipse(this.position.x, this.position.y, this.boundingRadius * 2);
        */
        // ---

        this.p.pop();
    }
}