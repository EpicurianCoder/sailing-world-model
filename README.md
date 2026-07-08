World Model Sailing Project

## Project Outline

The objective is to create a simple world model that train a model to control and operate a simple sailboat.

The world model will be as simple as possible and consist of three elements (Vision, Memory and Controller). The environemnt will be a 256 x 256 2D top down grid, represent a section of Ocean. For each run, the environment will be procedurally geenrated, and a number of land masses will be added to the environment.

A small boat will start each simulation at the bottom centre of the environment, at approxiamately (128, 10). In every iteration, the boat will attempt to navigate the environment, avoiding any land masses, and cross the northern most edge of the map, anywhere between the points (64, 256) and (192, 256).

Each simulation round will have a randomly generated wind direction, and wind velocity that will remain constant for the duration of each simulation.

The generated land masses will never fully block a path between the boat and the objective location, and will never make channels between land masses with gaps smaller than the minimum I will define.

The boat has the ability to change it sail angle and size, and to change its rudder angle (This will be simplified to the boat simply turning). These actions can each be taken at a rate of X degrees per frame/step in order to prevent instant unrealistic large changes.

The speed of the boat will be calculated using the traditional formula for how much the angle of the sail strays from the optimal angle to capture the wind, according the wind direction and the boat angle, and then we will also take drag into account, but keep the other advanced forces and mechanics to a minumum for the sake of simplicity.

## The Boat

The boat has two methods of control it can affect, this is the turning of itself (the rudder) and the changin of the angle of the sail and the ability to increase and reduce its sail size, to better capture and avoid the wind when changin direction into or against the wind, and wanting to avoid caturing the wind as much.

The boat will graphically be shown as having a of 7 pixels long, and one pixel wide. The sail will shown to be 6 pixels long (at maximum) when fully 'extended' and will be 3 pixel long (minimum) when 'retracted', and begin at the second pixel on the boat, which will act as the pivot point.

Other than gaining and losing speed, due to the focres calcuated by the wind acting on the sail, the boat will only experience drag, and no other advance forces will act on the boat.

## The Environment

### The Physics Engine

The physics engine balances computational simplicity with authentic sailing mechanics. It handles upwind tacking, backward drifting, aerodynamic drag, and directional capsizing without requiring computationally expensive fluid dynamics.

### The Map

The Map will be procedurally generated and will be a simple 'empty' grid, representing the ocean, and organic shapes that populate this space, representing land masses.

#### Rules for landmasses.

- Randomly generated number of landmasses, between 0 and 5.
- Any one is never large than 1/8 of the map.
- The total landmass volume never exceeds 1/3 of the map.
- Any gap between any two edges of different islands is never smaller than 48 pixels.
- Simple math formula is used to draw orangic shapes, nothing extreme.

### The Wind

- Random for each simulation/iteration
- Constant for the entire duration of the simulation
- Has a direction Value
- Has a strength Value
- Is indicated on the map by a red arrow in the top left corner, occupying a box 16x16, and inset 1 pixel from the top and the edge.

## Project Variables

At any given time, the following values should exist within the simulation.

- Wind Direction
- Wind Strength
- Boat Direction
- Boat Speed (Velocity)
- Sail Angle
- Sail Size
- Boat location (X, Y coordinates: the central pixel on the 7-pixel boat visual)

## Desired Outcomes

- The boat iteratively learns to captured the wind and progress towards it objective
- The boat learns to deal with a variety of wind directions
- The boat becomes able to travel upwind
- The boat learns to use the ability to change sail angle and size sail to optimize capturing the wind.
- The system uses the wind direction and boat direction to calculate the optimal sail angle at each step, and then calculates how much the sail angle differs from this angle, and works out the continual force applied to the boat, to determine whether its speed will increase of decrease in teh subsequent step, and how drag effect this.
