import controlP5.*;
PImage bgImage;
ControlP5 cp5;
ParticleSystem ps;

boolean isSimulationRunning = false;
int particleCount = 1000;
float viscosity = 0.02; 
float speed = 2.0; 

// Box boundaries
int boxX = 350;
int boxY = 200;
int boxWidth = 800;
int boxHeight = 500;

PVector wind = new PVector(0, 0); // Wind vector initialized with no wind
boolean windEnabled = false; // Wind effect toggle

int windStartTime = 0; // Tracks when the wind was enabled
float temperature = 10.0; // Default temperature in degrees Celsius

int coolColor = color(0, 0, random(100,255)); // Blue for cool
int warmColor = color(random(100,255), 0, 0); // Red for warm
boolean addingObstacle = false;

void setup() {
  fullScreen();
  ps = new ParticleSystem();
  PFont font = createFont("Arial-Bold", 16);
  bgImage = loadImage("bg.jpeg");
  int backgroundColor = color(10, 105, 150);
  int sliderColor = color(180, 180, 220);
  int textColor = color(255, 255, 255);

  background(backgroundColor);

  cp5 = new ControlP5(this);
  
  // Slider for particle count
  cp5.addSlider("particle Count")
     .setPosition(50, 100)
     .setSize(20,20)
     .setWidth(200)
     .setRange(10, 5000)
     .setValue(1000)
   .getCaptionLabel()
     .align(ControlP5.LEFT, ControlP5.TOP_OUTSIDE) 
     ;
     
  // Slider for viscosity
  cp5.addSlider("viscosity")
     .setPosition(50, 150)
     .setWidth(200)
     .setSize(200,20)
     .setRange(0.01, 0.1)
     .setValue(0.03)
     .getCaptionLabel()
     .align(ControlP5.LEFT, ControlP5.TOP_OUTSIDE) 
     ;
     
  // Slider for speed
  cp5.addSlider("speed")
     .setPosition(50, 200)
     .setWidth(200)
     .setSize(200,20)
     .setRange(1, 5)
     .setValue(2)
     .getCaptionLabel()
     .align(ControlP5.LEFT, ControlP5.TOP_OUTSIDE) 
     ;
     
     cp5.addButton("toggleWind")
   .setPosition(50, 250)
   .setSize(100, 20)
   .setLabel("Wind OFF");

  // Add a temperature slider
  cp5.addSlider("temperature")
   .setPosition(50, 300)
   .setWidth(200)
   .setSize(200,20)
   .setRange(0, 40)
   .setValue(10)
   .getCaptionLabel()
   .align(ControlP5.LEFT, ControlP5.TOP_OUTSIDE);

  cp5.addButton("addObstacle")
  .setPosition(50, 350)
  .setSize(100, 30)
  .setLabel("Add Obstacle")
  .onClick(new CallbackListener() {
    public void controlEvent(CallbackEvent event) {
      addingObstacle = !addingObstacle; 
    }
  });

  cp5.addButton("resetObstacles")
    .setPosition(50, 400) 
    .setSize(100, 20)
    .setLabel("Reset Obstacles")
    .onClick(new CallbackListener() {
      public void controlEvent(CallbackEvent event) {
        ps.clearObstacles();
      }
    });
}

void restartSimulation() {
  ps = new ParticleSystem();
  isSimulationRunning = true; 
}

void mouseClicked() {
 int buttonHeight = 50;
  int playPauseButtonWidth = 100;
  int playPauseButtonX = width / 2 - playPauseButtonWidth - 10; 
  int restartButtonWidth = 100;
  int restartButtonX = width / 2 + 10; 
  int buttonY = 100; 

  // Play/Pause button click area
  if (mouseX > playPauseButtonX && mouseX < playPauseButtonX + playPauseButtonWidth && mouseY > buttonY && mouseY < buttonY + buttonHeight) {
    isSimulationRunning = !isSimulationRunning; // Toggle the running state
  }
  
  // Restart button click area
  else if (mouseX > restartButtonX && mouseX < restartButtonX + restartButtonWidth && mouseY > buttonY && mouseY < buttonY + buttonHeight) {
    restartSimulation(); // Call the method to restart the simulation
  }
  
  // Check if we are in adding obstacle mode and the click is within the main box
  if (addingObstacle && mouseX > boxX && mouseX < boxX + boxWidth &&
      mouseY > boxY && mouseY < boxY + boxHeight) {
      
    // Predefined obstacle size
    int obstacleWidth = 100;
    int obstacleHeight = 50;
    
    // Adjust the position to prevent the obstacle from extending outside the box
    int obstacleX = constrain(mouseX - obstacleWidth / 2, boxX, boxX + boxWidth - obstacleWidth);
    int obstacleY = constrain(mouseY - obstacleHeight / 2, boxY, boxY + boxHeight - obstacleHeight);
    
    // Add the obstacle
    Obstacle newObstacle = new Obstacle(obstacleX, obstacleY, obstacleWidth, obstacleHeight);
    ps.addObstacle(newObstacle);

    // turn off addingObstacle mode after adding one obstacle
     addingObstacle = false;
      }

}
void drawPlayPauseButton() {
  int playPauseButtonWidth = 100;
  int buttonHeight = 50;
  int playPauseButtonX = width / 2 - playPauseButtonWidth - 10; 
  int restartButtonWidth = 100;
  int restartButtonX = width / 2 + 10; 
  int buttonY = 100; 
  
  // Play/Pause Button
  fill(255, 0, 0); // Red button
  noStroke();
  rect(playPauseButtonX, buttonY, playPauseButtonWidth, buttonHeight); 
  fill(255); // White text for Play/Pause
  textSize(20);
  textAlign(CENTER, CENTER);
  text(isSimulationRunning ? "Pause" : "Play", playPauseButtonX + playPauseButtonWidth / 2, buttonY + buttonHeight / 2);

  // Restart Button
  fill(0, 0, 255); // Blue button for Restart
  rect(restartButtonX, buttonY, restartButtonWidth, buttonHeight); 
  fill(255); // White text for Restart
  text("Restart", restartButtonX + restartButtonWidth / 2, buttonY + buttonHeight / 2);
  
 
}

void toggleWind() {
    windEnabled = !windEnabled;
    if (windEnabled) {
        wind.set(0.3, 0); // wind direction to the right
        windStartTime = millis(); // Record the time when wind was enabled
        cp5.get(Button.class, "toggleWind").setLabel("Wind ON"); // Update button label on
    } else {
        wind.set(0, 0); // No wind
        cp5.get(Button.class, "toggleWind").setLabel("Wind OFF"); // Update button label off
    }
}

void drawFish() {
  int fishX = boxX - 80; // Position fish to the left of the box
  int fishY = boxY + boxHeight / 2; 
  
  fill(255, 200, 0);
  noStroke();
  
  // Fish body
  ellipse(fishX, fishY, 120, 60);
  
  // Fish tail
  triangle(fishX - 60, fishY, fishX - 100, fishY - 30, fishX - 100, fishY + 30);
  
  // Fish eye
  fill(0);
  ellipse(fishX + 40, fishY - 15, 10, 10);
  
  // Fish mouth
  stroke(0);
  strokeWeight(2);
  line(fishX + 60, fishY, fishX + 75, fishY);
  noStroke();
}

void drawGradientBackground() {
  int topColor, bottomColor;
  float tempFactor = map(temperature, 0, 40, 0, 1);
  
  topColor = lerpColor(color(100, 100, 200), color(255, 188, 100), tempFactor); // Light sky blue to light peach
  bottomColor = lerpColor(color(130, 200, 180), color(255, 100, 100), tempFactor); // Powder blue to red

 
  for (int i = boxY; i <= boxY + boxHeight; i++) {
    float inter = map(i, boxY, boxY + boxHeight, 0, 1);
    int c = lerpColor(topColor, bottomColor, inter); // Deep sky blue gradient
    stroke(c);
    line(boxX, i, boxX + boxWidth, i);
  }
}



void draw() {
 image(bgImage, 0, 0, width, height);
  drawGradientBackground();
   drawFish();
   
  // Check if wind has been on for more than 5 seconds
  if (windEnabled && millis() - windStartTime > 5000) {
      windEnabled = false;
      wind.set(0, 0); // Stop the wind
      cp5.get(Button.class, "toggleWind").setLabel("Wind OFF");
  }
  
  stroke(0);
  noFill();
  rect(boxX, boxY, boxWidth, boxHeight);
   drawPlayPauseButton();
  for (Obstacle obstacle : ps.obstacles) {
    obstacle.display();
  }
  if (isSimulationRunning) {
  ps.run();
  ps.adjustParticleCount(particleCount);
   for (Obstacle obstacle : ps.obstacles) {
    obstacle.display();
  }
  for (Particle p : ps.particles) {
    p.checkObstacleCollision(ps.obstacles);
    p.update();
    p.display();
  }
  }

}



class Particle {
  PVector position;
  PVector velocity;
  float radius = 7;
  int particleColor; 
  PVector gravity = new PVector(0, 1); // Gravity force vector pointing down
  float lifespan = random(1000, 2000);; 
   
  Particle() {
    reset();
  }
  

  void reset() {
      // Initialize position near the fish's mouth
      this.position = new PVector(boxX +10, boxY + boxHeight / 2);
      // Set initial velocity to simulate flow from the fish's mouth into the box
       this.velocity = new PVector(random(-0.5, 0.5), random(-0.5, 0.5)); 
      this.lifespan = random(1000, 2000); // Reset lifespan
      updateColorBasedOnTemperature();
      
  }
  void updateColorBasedOnTemperature() {
      // Map the temperature to a value between 0 and 1 for lerpColor
      float tempFactor = map(temperature, 0, 40, 0, 1);
       int coolColor = color(0, 0, random(100,255)); // Randomized cool color
      int warmColor = color(random(100,255), 0, 0); // Randomized warm color
      particleColor = lerpColor(coolColor, warmColor, tempFactor);
    }
  
  void display() {
    fill(particleColor);
    noStroke();
    ellipse(position.x, position.y, radius * 2, radius * 2);
  }

  void adjustLifespanBasedOnTemperature() {
      // Lifespan decreases as temperature increases
      // At lower temperatures (0 degrees), lifespan is longer
      // At higher temperatures (40 degrees), lifespan is shorter

    float baseLifespan = 5000; // Base lifespan at lowest temperature
    float minLifespan = 500; // Minimum lifespan at highest temperature 
    
    float temperatureFactor = pow(temperature / 40, 2); // Squared to emphasize the reduction at higher temperatures
    
    // Calculate new lifespan, ensuring it doesn't go below the minimum lifespan
    this.lifespan = max(minLifespan, baseLifespan * (1 - temperatureFactor));
      
  }

  void bounceOffWalls() {
    // Handle horizontal boundaries
    if (position.x - radius <= boxX) {
        position.x = boxX + radius;
        velocity.x *= -1;
    } else if (position.x + radius >= boxX + boxWidth) {
        position.x = boxX + boxWidth - radius;
        velocity.x *= -1;
    }

    // Handle bottom boundary with energy loss
    if (position.y + radius >= boxY + boxHeight) {
        position.y = boxY + boxHeight - radius;
        velocity.y *= -0.5; // Simulate a loss of energy upon hitting the bottom
    }
     if (position.y - radius <= boxY) {
      position.y = boxY + radius;
      velocity.y *= -1; // Reflect the velocity so the particle bounces downward
    }
  }

  void applyForce(PVector force) {
    velocity.add(force);
  }

  void checkObstacleCollision(ArrayList<Obstacle> obstacles) {
    for (Obstacle obs : obstacles) {
      // Predict next position
      PVector nextPosition = PVector.add(position, velocity);
  
      // Check for collision with this obstacle
      if (nextPosition.x + radius > obs.x && nextPosition.x - radius < obs.x + obs.width &&
          nextPosition.y + radius > obs.y && nextPosition.y - radius < obs.y + obs.height) {
        
        // Determine collision direction
        boolean collisionX = position.x + radius > obs.x && position.x - radius < obs.x + obs.width;
        boolean collisionY = position.y + radius > obs.y && position.y - radius < obs.y + obs.height;
  
        // Adjust velocity based on collision direction
        if (collisionX && !collisionY) {
          velocity.y *= -1;
        } else if (!collisionX && collisionY) {
          velocity.x *= -1;
        } else {
          velocity.x *= -1;
          velocity.y *= -1;
        }
  
        // Move particle outside the obstacle 
        while (position.x + radius >= obs.x && position.x - radius <= obs.x + obs.width &&
               position.y + radius > obs.y && position.y - radius < obs.y + obs.height) {
          position.add(velocity);
        }
      }
    }
  }


  void interact(ArrayList<Particle> particles) {
      for (Particle other : particles) {
          if (other != this) {
              float d = PVector.dist(position, other.position);
              float perceptionRadius = radius * 2; // Distance at which particles are touching each other
              if (d < perceptionRadius) {
                  // Adjust positions to prevent overlap
                  PVector correctionVector = PVector.sub(position, other.position);
                  correctionVector.setMag((perceptionRadius - d) / 2); // Half the overlap so both particles adjust equally
                  position.add(correctionVector);
                  other.position.sub(correctionVector);
  
                  velocity.y *= -0.5; // A simple response that simulates a loss of energy upon collision
                  other.velocity.y *= -0.5;
              }
              
          }
      }
  }

  void update() {
      // Apply gravity and wind (if enabled) to all particles
      if (windEnabled) {
          velocity.add(wind);
      }
      velocity.add(gravity);
      
      velocity.add(PVector.random2D().mult(random(0.5) * speed)); 
      // Apply viscosity as damping and update position
      velocity.mult(1 - viscosity);
      position.add(velocity);
  
      // Check for collision with floor and adjust position and velocity
      bounceOffWalls();
      float tempEffect = map(temperature, 0, 40, 1, 3); // At 40 degrees, lifespan decreases 3 times faster
      lifespan -= tempEffect; // Adjust this rate as needed
          
      
      if (lifespan <= 0) {
          reset(); // Reset the particle if its lifespan has expired
      }
      for (Obstacle obstacle : ps.obstacles) {
      if (obstacle.contains(position)) {
        velocity.mult(-1); // Simple reflection for demonstration
      }
    }
     
  }


  void repelFromMouse() {
    float mouseRepelRadius = 100; // Distance within which the mouse repels particles
    PVector mousePos = new PVector(mouseX, mouseY);
    float d = PVector.dist(position, mousePos);
    
    if (d < mouseRepelRadius) {
        PVector repelForce = PVector.sub(position, mousePos); // Direction from mouse to particle
        float strength = 1 - (d / mouseRepelRadius); // Stronger force when closer to the mouse
        repelForce.setMag(strength * 0.5); // Adjust magnitude of the repel force
        velocity.add(repelForce);
    }
  }

  void mouseDragged() {
    cp5.get(Button.class, "toggleWind").setLabel("Wind OFF");
    wind.set((mouseX - pmouseX) * 0.05, (mouseY - pmouseY) * 0.05); // Create wind based on mouse drag
    windEnabled = true;
    windStartTime = millis(); // Reset wind timer on drag
    }
}
  
void controlEvent(ControlEvent event) {
  if (event.isFrom("particle Count")) {
    particleCount = (int) event.getValue();
    ps.adjustParticleCount(particleCount);
  }
  if (event.isFrom("temperature")) {
    temperature = event.getValue();
     for (Particle particle : ps.particles) {
            particle.updateColorBasedOnTemperature();
        }
  }
}

class ParticleSystem {
  ArrayList<Particle> particles = new ArrayList<Particle>();

  ParticleSystem() {
    for (int i = 0; i < particleCount; i++) {
      particles.add(new Particle());
    }
  }
  ArrayList<Obstacle> obstacles = new ArrayList<Obstacle>();

  void addObstacle(Obstacle obstacle) {
    obstacles.add(obstacle);
  }  

void clearObstacles() {
    obstacles.clear(); // Clear the list of obstacles
  }

  void run() {
    for (int i = particles.size() - 1; i >= 0; i--) {
    Particle p = particles.get(i);
    p.interact(particles);
    p.repelFromMouse();
    p.update();
    p.display();    
    }
  }

  void adjustParticleCount(int newCount) {
    while (particles.size() < newCount) {
      particles.add(new Particle());
    }
    while (particles.size() > newCount && particles.size() > 0) {
      particles.remove(particles.size() - 1);
    }
  }
}

class Obstacle {
  int x, y, width, height;
  
  Obstacle(int x, int y, int width, int height) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }
  
  void display() {
    fill(120, 50, 50); 
    noStroke();
    rect(x, y, width, height);
  }
  
  boolean contains(PVector point) {
    return point.x >= x && point.x <= x + width && point.y >= y && point.y <= y + height;
  }
}
