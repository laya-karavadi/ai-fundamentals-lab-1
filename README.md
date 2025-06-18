# AI Fundamentals Lab 1: Optimization Algorithms

A Python implementation exploring three fundamental optimization algorithms: Hill Climbing, Random Restart Hill Climbing, and Simulated Annealing. This lab demonstrates how different search strategies can be used to find optimal solutions in complex search spaces.

## 🎯 Objective

Find the maximum value of a complex 2D objective function using various optimization algorithms. The objective function represents a mathematical landscape with multiple peaks and valleys, simulating real-world optimization problems.

## 📊 The Objective Function

The lab uses a complex mathematical function that creates a challenging optimization landscape:

```python
def objective_function(X):
    x = X[0]
    y = X[1]
    value = 3 * (1 - x) ** 2 * math.exp(-x ** 2 - (y + 1) ** 2) - \
            10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-x ** 2 - y ** 2) - \
            (1 / 3) * math.exp(-(x + 1) ** 2 - y ** 2)
    return value
```

### Function Characteristics:
- **Domain**: x ∈ [-4, 4], y ∈ [-4, 4]
- **Multiple local maxima**: Creates a challenging search space
- **Continuous**: Smooth gradients allow for step-wise optimization
- **Non-convex**: Traditional gradient methods may get stuck in local optima

## 🚀 Algorithms Implemented

### 1. Hill Climbing Algorithm

A simple local search algorithm that continuously moves toward higher values.

#### Features:
- **Greedy approach**: Always moves to the best neighboring state
- **Four-directional search**: Explores neighbors in x and y directions
- **Step-by-step optimization**: Uses configurable step size
- **Early termination**: Stops when no better neighbor is found

#### Parameters:
- `initial_state`: Starting point (default: [0, 0])
- `step_size`: Movement increment (default: 0.01)
- `print_iters`: Debug output toggle

#### Limitations:
- **Local optima**: Can get stuck at the first peak it encounters
- **No backtracking**: Cannot escape local maxima

### 2. Random Restart Hill Climbing

An enhanced version that runs hill climbing multiple times from different starting points.

#### Features:
- **Multiple attempts**: Runs hill climbing from random starting positions
- **Global exploration**: Reduces chance of missing the global optimum
- **Best solution tracking**: Keeps track of the best result across all restarts

#### Parameters:
- `num_restarts`: Number of random starting points (default: 10)
- `lower_bounds`: Minimum x, y values
- `upper_bounds`: Maximum x, y values
- `step_size`: Movement increment

#### Advantages:
- **Better global search**: Higher probability of finding global optimum
- **Robustness**: Less sensitive to initial conditions

### 3. Simulated Annealing

A probabilistic optimization algorithm inspired by the metallurgical annealing process.

#### Features:
- **Probabilistic acceptance**: Can accept worse solutions with decreasing probability
- **Temperature schedule**: Controls exploration vs. exploitation trade-off
- **Escape mechanism**: Can escape local optima early in the search
- **Convergence**: Gradually becomes more greedy as temperature decreases

#### Parameters:
- `initial_temp`: Starting temperature (default: 1000)
- `cooling_rate`: Temperature reduction factor (default: 0.99)
- `min_temp`: Stopping temperature (default: 1e-3)
- `step_size`: Random step magnitude (default: 0.1)

#### Key Mechanism:
```python
# Accept new state if better OR with probability based on temperature
if delta > 0 or np.random.rand() < np.exp(delta / temperature):
    current_state = new_state
```

## 🔧 Implementation Details

### Hill Climbing Enhanced Features

The implementation includes an **Extra Credit Enhancement**: First-choice hill climbing variant that randomly shuffles neighbors and accepts the first improvement found, rather than evaluating all neighbors.

```python
# First-choice hill climbing enhancement
np.random.shuffle(neighbors)
for neighbor in neighbors:
    neighbor_value = objective_function(neighbor)
    if neighbor_value > current_value:
        best_neighbor = neighbor
        best_value = neighbor_value
        break
```

### Boundary Handling

All algorithms ensure solutions remain within the defined search space:
- Hill climbing: Implicit boundary respect through neighbor generation
- Simulated annealing: Explicit clipping using `np.clip()`

## 📈 Expected Results

### Algorithm Performance Comparison:

1. **Hill Climbing**: 
   - Fast convergence
   - May find local optima
   - Deterministic given same starting point

2. **Random Restart Hill Climbing**:
   - Better global search capability
   - Higher computational cost
   - More consistent results across runs

3. **Simulated Annealing**:
   - Best balance of exploration and exploitation
   - Can escape local optima
   - Stochastic but generally reliable

## 🛠️ Usage

### Prerequisites
```python
import numpy as np
import math
```

### Basic Execution

```python
# Define the search space
lower_bounds = [-4, -4]
upper_bounds = [4, 4]

# Run Hill Climbing
hill_climbing_solution = hill_climbing(objective_function)
print('Hill climbing solution:', hill_climbing_solution)

# Run Random Restart Hill Climbing
random_restart_solution = random_restart_hill_climbing(
    objective_function, lower_bounds, upper_bounds, num_restarts=10
)
print('Random restart solution:', random_restart_solution)

# Run Simulated Annealing
simulated_annealing_solution = simulated_annealing(
    objective_function, lower_bounds, upper_bounds
)
print('Simulated annealing solution:', simulated_annealing_solution)
```

### Parameter Tuning

#### Hill Climbing:
```python
# More precise search
solution = hill_climbing(objective_function, step_size=0.001)

# Different starting point
solution = hill_climbing(objective_function, initial_state=np.array([1, 1]))
```

#### Random Restart:
```python
# More thorough search
solution = random_restart_hill_climbing(
    objective_function, lower_bounds, upper_bounds, 
    num_restarts=50, step_size=0.005
)
```

#### Simulated Annealing:
```python
# Slower cooling for better exploration
solution = simulated_annealing(
    objective_function, lower_bounds, upper_bounds,
    initial_temp=2000, cooling_rate=0.995, step_size=0.2
)
```

## 🧪 Experimental Analysis

### Key Questions to Explore:

1. **Convergence**: Which algorithm finds the best solution most consistently?
2. **Efficiency**: What's the trade-off between computation time and solution quality?
3. **Robustness**: How sensitive is each algorithm to parameter changes?
4. **Exploration vs. Exploitation**: How well does each algorithm balance these competing needs?

### Suggested Experiments:

1. **Multiple Runs**: Execute each algorithm 20+ times and analyze result distributions
2. **Parameter Sensitivity**: Test different step sizes, restart counts, and cooling schedules
3. **Starting Point Analysis**: Compare performance with different initial conditions
4. **Convergence Plots**: Track objective function value over iterations

## 📚 Learning Outcomes

This lab demonstrates fundamental concepts in AI optimization:

- **Local vs. Global Search**: Understanding the limitations of greedy approaches
- **Stochastic Methods**: How randomness can improve search performance
- **Parameter Tuning**: The importance of algorithm configuration
- **Trade-offs**: Balancing computation time vs. solution quality
- **Algorithm Selection**: Choosing the right tool for different problem types

## 🔍 Extensions and Improvements

### Potential Enhancements:

1. **Adaptive Step Size**: Dynamically adjust step size based on progress
2. **Population-Based Methods**: Implement genetic algorithms or particle swarm optimization
3. **Hybrid Approaches**: Combine multiple algorithms for better performance
4. **Visualization**: Plot the objective function landscape and search trajectories
5. **Multi-Modal Analysis**: Develop methods to find multiple good solutions

### Advanced Features:
```python
# Adaptive simulated annealing
def adaptive_simulated_annealing(objective_function, lower_bounds, upper_bounds):
    # Adjust step size based on acceptance rate
    # Implement adaptive cooling schedule
    # Add restart mechanism when stuck
    pass
```

## 📊 Performance Metrics

### Evaluation Criteria:
- **Best Value Found**: Maximum objective function value achieved
- **Convergence Speed**: Number of iterations to reach near-optimal solution
- **Consistency**: Standard deviation across multiple runs
- **Success Rate**: Percentage of runs finding global optimum (within tolerance)

---

This lab provides hands-on experience with fundamental optimization algorithms used throughout artificial intelligence and machine learning. Understanding these basic search strategies is crucial for tackling more complex AI problems involving parameter optimization, neural network training, and automated decision-making.
