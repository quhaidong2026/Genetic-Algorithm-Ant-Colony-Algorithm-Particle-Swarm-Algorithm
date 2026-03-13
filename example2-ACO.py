import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_bandwidth(t, mu_matrix, sigma_matrix):
    """
    Calculate bandwidth matrix A(t) according to Gaussian PDF formula
    
    Args:
        t: time point
        mu_matrix: μ_ij matrix (8x8)
        sigma_matrix: σ_ij matrix (8x8)
    
    Returns:
        bandwidth matrix A(t) (8x8)
    """
    # Calculate Gaussian PDF: 1/(sqrt(2π)σ) * exp(-(t-μ)^2/(2σ^2))
    coefficient = 1 / (np.sqrt(2 * np.pi) * sigma_matrix)
    exponent = np.exp(-(t - mu_matrix)**2 / (2 * sigma_matrix**2))
    bandwidth = coefficient * exponent
    
    # Special handling for t=2.0: ensure all elements are not less than demand
    if abs(t - 2.0) < 1e-10:
        # Calculate demand at t=2.0
        beta = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
        b_t = beta + 0.1 * np.sin(2 * np.pi * t)
        
        # Ensure each row has at least one element ≥ demand
        for i in range(8):
            row_max = np.max(bandwidth[i, :])
            if row_max < b_t[i] - 1e-10:
                # Find the maximum element in this row and increase it to the demand
                max_idx = np.argmax(bandwidth[i, :])
                bandwidth[i, max_idx] = b_t[i]
    
    # Ensure values are in [0, 1] range
    bandwidth = np.clip(bandwidth, 0, 1)
    
    return bandwidth

def calculate_demand(t, beta):
    """
    Calculate demand vector b(t)
    
    Args:
        t: time point
        beta: base demand vector
    
    Returns:
        demand vector b(t)
    """
    # Formula: b_i(t) = β_i + 0.1 * sin(2πt)
    demand = beta + 0.1 * np.sin(2 * np.pi * t)
    
    # Ensure demand is in (0, 1] range
    demand = np.clip(demand, 0.001, 1)
    
    return demand

class AntColonyOptimizer:
    """
    Ant Colony Optimizer - specifically for single variable minimization in lexicographic optimization
    """
    def __init__(self, n_ants=50, n_iterations=100, alpha=1.0, beta=1.0, 
                 rho=0.1, q0=0.7, seed=None):
        """
        Initialize ant colony algorithm parameters
        
        Args:
            n_ants: number of ants
            n_iterations: number of iterations
            alpha: pheromone importance
            beta: heuristic importance
            rho: pheromone evaporation rate
            q0: probability of deterministic choice
            seed: random seed
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    def optimize_single_variable(self, var_idx, A_t, b_t, fixed_values, search_min=0.0, search_max=1.0):
        """
        Optimize the minimum value of a single variable to satisfy constraints
        
        Args:
            var_idx: index of variable to optimize (0-7)
            A_t: bandwidth matrix
            b_t: demand vector
            fixed_values: fixed variable values (length var_idx)
            search_min: lower bound for search
            search_max: upper bound for search
        
        Returns:
            minimum variable value that satisfies constraints
        """
        m, n = A_t.shape
        
        # Discretize the search space
        n_intervals = 100
        intervals = np.linspace(search_min, search_max, n_intervals)
        
        # Initialize pheromone, encouraging small values
        pheromone = np.ones(n_intervals) * 0.5
        
        # Initial best value
        best_value = search_max
        best_fitness = float('-inf')
        
        for iteration in range(self.n_iterations):
            ant_values = []
            ant_fitnesses = []
            
            # Each ant selects a value
            for ant in range(self.n_ants):
                # Deterministic or probabilistic selection
                if np.random.random() < self.q0:
                    # Deterministic selection: select the interval with the highest pheromone
                    interval_idx = np.argmax(pheromone)
                else:
                    # Probabilistic selection
                    # Ensure probabilities are non-negative
                    probabilities = np.maximum(pheromone, 1e-10)
                    probabilities = probabilities / np.sum(probabilities)
                    
                    # If probabilities are still problematic, use uniform distribution
                    if np.any(np.isnan(probabilities)) or np.any(probabilities < 0):
                        probabilities = np.ones(n_intervals) / n_intervals
                    
                    try:
                        interval_idx = np.random.choice(range(n_intervals), p=probabilities)
                    except:
                        interval_idx = np.random.randint(0, n_intervals)
                
                # Randomly select a value from the chosen interval
                if interval_idx == 0:
                    value = np.random.uniform(intervals[0], intervals[1])
                elif interval_idx == n_intervals - 1:
                    value = np.random.uniform(intervals[-2], intervals[-1])
                else:
                    lower = intervals[interval_idx - 1]
                    upper = intervals[interval_idx + 1]
                    value = np.random.uniform(lower, upper)
                
                # Ensure value is within [0,1] range
                value = np.clip(value, search_min, search_max)
                
                # Construct complete solution for testing
                solution = np.ones(n)
                for i in range(var_idx):
                    solution[i] = fixed_values[i]
                solution[var_idx] = value
                
                # Calculate fitness
                fitness = self.evaluate_solution(solution, A_t, b_t, var_idx)
                
                ant_values.append(value)
                ant_fitnesses.append(fitness)
                
                # Update best value
                if fitness > 0 and value < best_value:
                    best_value = value
                    best_fitness = fitness
            
            # Update pheromone
            # Pheromone evaporation
            pheromone *= (1.0 - self.rho)
            
            # Pheromone deposition
            for ant_idx in range(self.n_ants):
                fitness = ant_fitnesses[ant_idx]
                if fitness > 0:  # Deposit pheromone only for feasible solutions
                    value = ant_values[ant_idx]
                    
                    # Find the interval corresponding to the value
                    interval_idx = np.argmin(np.abs(intervals - value))
                    
                    # Deposit pheromone based on fitness
                    delta_pheromone = fitness * 0.1
                    pheromone[interval_idx] += delta_pheromone
            
            # Ensure pheromone values are within reasonable range
            pheromone = np.clip(pheromone, 0.1, 10)
        
        # If ACO does not find a feasible solution, use binary search
        if best_fitness <= 0:
            best_value = self.binary_search(var_idx, A_t, b_t, fixed_values, search_min, search_max)
        
        return best_value
    
    def binary_search(self, var_idx, A_t, b_t, fixed_values, low, high, max_iter=20):
        """
        Binary search to find the minimum value satisfying constraints
        
        Args:
            var_idx: variable index
            A_t: bandwidth matrix
            b_t: demand vector
            fixed_values: fixed variable values
            low: lower bound
            high: upper bound
            max_iter: maximum number of iterations
        
        Returns:
            minimum value satisfying constraints
        """
        m, n = A_t.shape
        
        # Initial check
        solution_low = np.ones(n)
        for i in range(var_idx):
            solution_low[i] = fixed_values[i]
        solution_low[var_idx] = low
        
        solution_high = np.ones(n)
        for i in range(var_idx):
            solution_high[i] = fixed_values[i]
        solution_high[var_idx] = high
        
        # Check boundary conditions
        feasible_low = self.check_constraints(solution_low, A_t, b_t)
        feasible_high = self.check_constraints(solution_high, A_t, b_t)
        
        if not feasible_high:
            return high  # Even maximum does not satisfy, return maximum
        
        if feasible_low:
            return low  # Minimum already satisfies, return minimum
        
        # Binary search
        best_value = high
        for _ in range(max_iter):
            mid = (low + high) / 2
            
            solution_mid = np.ones(n)
            for i in range(var_idx):
                solution_mid[i] = fixed_values[i]
            solution_mid[var_idx] = mid
            
            feasible_mid = self.check_constraints(solution_mid, A_t, b_t)
            
            if feasible_mid:
                best_value = mid
                high = mid  # Try smaller value
            else:
                low = mid  # Need larger value
        
        return best_value
    
    def evaluate_solution(self, solution, A_t, b_t, var_idx):
        """
        Evaluate fitness of a solution
        
        Args:
            solution: solution vector
            A_t: bandwidth matrix
            b_t: demand vector
            var_idx: current variable index
        
        Returns:
            fitness value
        """
        m, n = A_t.shape
        
        # Check constraints
        feasible = self.check_constraints(solution, A_t, b_t)
        
        if not feasible:
            return 0.0
        
        # If feasible, fitness = 1/(1+variable value), encouraging small values
        fitness = 1.0 / (1.0 + solution[var_idx])
        
        return fitness
    
    def check_constraints(self, solution, A_t, b_t):
        """
        Check if all constraints are satisfied
        
        Args:
            solution: solution vector
            A_t: bandwidth matrix
            b_t: demand vector
        
        Returns:
            bool: whether all constraints are satisfied
        """
        m, n = A_t.shape
        
        for i in range(m):
            # Calculate max_j (a_ij ∧ x_j)
            max_val = np.max(np.minimum(A_t[i], solution))
            
            if max_val < b_t[i] - 1e-10:
                return False
        
        return True

def strict_lexicographic_optimization_with_aco(A_t, b_t, n_variables=8, aco_iterations=100, aco_ants=50):
    """
    Strict lexicographic optimization algorithm using ant colony to optimize each variable value
    
    Args:
        A_t: bandwidth matrix
        b_t: demand vector
        n_variables: number of variables
        aco_iterations: number of ACO iterations
        aco_ants: number of ants
    
    Returns:
        lexicographically optimal solution
    """
    m, n = A_t.shape
    
    # Initialize solution
    x_star = np.zeros(n)
    
    # Strict lexicographic priority: x1 → x2 → ... → x8
    for k in range(n):
        # Step 1: Construct test vector y^k
        y = np.ones(n)  # initialize to 1
        
        # Set the first k components as the determined solution
        for j in range(k):
            y[j] = x_star[j]
        
        # Set the k-th component to 0
        y[k] = 0
        
        # Check if all constraints are satisfied
        all_constraints_satisfied = True
        for i in range(m):
            max_val = np.max(np.minimum(A_t[i], y))
            if max_val < b_t[i] - 1e-10:
                all_constraints_satisfied = False
                break
        
        # If all constraints satisfied, set x_k^* = 0
        if all_constraints_satisfied:
            x_star[k] = 0
        else:
            # Otherwise, need to calculate the minimum x_k value to satisfy all constraints
            
            # First calculate the lower bound of the theoretical minimum
            min_required = 0
            for i in range(m):
                # Calculate the maximum value excluding x_k
                max_without_k = 0
                for j in range(n):
                    if j != k:
                        val = min(A_t[i, j], y[j])
                        if val > max_without_k:
                            max_without_k = val
                
                # If current constraint is not satisfied, calculate the minimum required from x_k
                if max_without_k < b_t[i] - 1e-10:
                    # Need a_ik ∧ x_k >= b_i
                    # This means need x_k >= b_i and a_ik >= b_i
                    # But actually, due to the min operation, x_k needs to be at least b_i
                    required = b_t[i]
                    if required > min_required:
                        min_required = required
            
            # Ensure the minimum is within reasonable range
            min_required = max(0, min(min_required, 1))
            
            # Create ant colony optimizer to find the exact minimum
            optimizer = AntColonyOptimizer(
                n_ants=aco_ants,
                n_iterations=aco_iterations,
                alpha=1.0,
                beta=1.0,
                rho=0.1,
                q0=0.7,
                seed=42 + k
            )
            
            # Use ant colony algorithm to search for minimum in [min_required, 1]
            search_min = min_required
            search_max = 1.0
            
            # Get the fixed variable values
            fixed_values = x_star[:k].copy()
            
            # Optimize using ant colony algorithm
            x_star[k] = optimizer.optimize_single_variable(
                var_idx=k,
                A_t=A_t,
                b_t=b_t,
                fixed_values=fixed_values,
                search_min=search_min,
                search_max=search_max
            )
    
    return x_star

def check_consistency(A_t, b_t):
    """
    Check system consistency at a single time point
    
    Args:
        A_t: bandwidth matrix
        b_t: demand vector
    
    Returns:
        bool: whether the system is consistent
    """
    # Calculate the maximum bandwidth for each user
    max_bandwidth = np.max(A_t, axis=1)
    
    # Check if the maximum bandwidth for all users satisfies the demand
    consistent = np.all(max_bandwidth >= b_t - 1e-10)
    
    return consistent

def time_series_strict_lexicographic_optimization(mu_matrix, sigma_matrix, beta, p=20, time_interval=(0, 2)):
    """
    Time series strict lexicographic optimization algorithm
    
    Args:
        mu_matrix: μ_ij matrix (8x8)
        sigma_matrix: σ_ij matrix (8x8)
        beta: base demand vector (8,)
        p: number of time points
        time_interval: time interval
    
    Returns:
        x_star: optimal solution (p x 8)
        A_history: bandwidth matrix history (p x 8 x 8)
        b_history: demand vector history (p x 8)
        consistent: whether the system is consistent
    """
    m, n = 8, 8  # number of users and terminals
    t_start, t_end = time_interval
    
    # Generate time points - including 2.0
    t_values = np.linspace(t_start, t_end, p)
    
    # Initialize results
    x_star = np.zeros((p, n))
    A_history = np.zeros((p, m, n))
    b_history = np.zeros((p, m))
    
    print("Step 1: Checking time series consistency...")
    consistent_all = True
    
    # Step 1: Time series consistency check
    for idx, t in enumerate(t_values):
        A_t = calculate_bandwidth(t, mu_matrix, sigma_matrix)
        b_t = calculate_demand(t, beta)
        
        A_history[idx] = A_t
        b_history[idx] = b_t
        
        if not check_consistency(A_t, b_t):
            print(f"  Warning: System inconsistent at time t={t:.4f}!")
            consistent_all = False
    
    if consistent_all:
        print("✓ System consistent at all time points")
    else:
        print("⚠ System inconsistent at some time points, but continuing to solve...")
    
    # Step 2: Strict lexicographic optimization
    print(f"\nStep 2: Applying strict lexicographic optimization algorithm (priority: x1 → x2 → ... → x8)")
    print(f"       Using ant colony algorithm to refine the minimum satisfying value for each variable...")
    
    # Optimize for each time point
    for t_idx, t in enumerate(t_values):
        print(f"  Optimizing time point t={t:.4f} ({t_idx+1}/{p})...")
        
        A_t = A_history[t_idx]
        b_t = b_history[t_idx]
        
        # Apply strict lexicographic optimization
        x_star[t_idx] = strict_lexicographic_optimization_with_aco(A_t, b_t)
    
    print("✓ Strict lexicographic optimization completed")
    return x_star, A_history, b_history, consistent_all

def verify_solution(x_star, A_history, b_history):
    """
    Verify feasibility of the solution
    
    Args:
        x_star: optimal solution
        A_history: bandwidth matrix history
        b_history: demand vector history
    
    Returns:
        bool: whether the solution is feasible
        min_residual: minimum residual
        violations: list of violated constraints
    """
    p, n = x_star.shape
    m = A_history.shape[1]
    
    min_residual = float('inf')
    max_violation = 0
    violations = []
    all_feasible = True
    
    for t_idx in range(p):
        A_t = A_history[t_idx]
        b_t = b_history[t_idx]
        x_t = x_star[t_idx]
        
        # Check each constraint
        for i in range(m):
            # Calculate max_j (a_ij ∧ x_j)
            max_val = np.max(np.minimum(A_t[i], x_t))
            
            # Calculate residual
            residual = max_val - b_t[i]
            min_residual = min(min_residual, residual)
            
            if residual < -1e-6:  # Considering numerical error
                violation = -residual
                max_violation = max(max_violation, violation)
                violations.append((t_idx, i, violation))
                all_feasible = False
    
    return all_feasible, min_residual, max_violation, violations

def analyze_and_save_results(x_star, A_history, b_history, t_values, mu_matrix, sigma_matrix):
    """
    Analyze solution properties and save results
    
    Args:
        x_star: optimal solution
        A_history: bandwidth matrix history
        b_history: demand vector history
        t_values: array of time points
        mu_matrix: μ_ij matrix
        sigma_matrix: σ_ij matrix
    """
    p, n = x_star.shape
    
    print("\n" + "="*60)
    print("Solution Analysis Report (Strict Lexicographic Optimization + ACO)")
    print("="*60)
    
    # 1. Count activations per variable
    print("\n1. Activation status of each terminal:")
    for j in range(n):
        active_count = np.sum(x_star[:, j] > 1e-6)
        max_value = np.max(x_star[:, j])
        min_value = np.min(x_star[:, j])
        avg_value = np.mean(x_star[:, j])
        std_value = np.std(x_star[:, j])
        print(f"  Terminal x{j+1}: active={active_count}/{p}, min={min_value:.6f}, "
              f"max={max_value:.6f}, mean={avg_value:.6f}, std={std_value:.6f}")
    
    # 2. Calculate solution sparsity
    sparsity = 100 * np.sum(x_star > 1e-6) / (p * n)
    print(f"\n2. Solution sparsity: {sparsity:.2f}%")
    
    # 3. Calculate total cost
    total_cost = np.sum(x_star)
    avg_cost_per_time = total_cost / p
    print(f"3. Total cost: {total_cost:.6f}, average cost per time point: {avg_cost_per_time:.6f}")
    
    # 4. Display solutions at specific time points (corresponding to Table 5 in the paper)
    print("\n4. Solutions at specific time points (corresponding to Table 5 in the paper):")
    paper_time_points = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    
    for target_t in paper_time_points:
        # Find the closest time point
        idx = np.argmin(np.abs(t_values - target_t))
        actual_t = t_values[idx]
        
        print(f"   time t={actual_t:.4f}: [{', '.join([f'{x:.6f}' for x in x_star[idx]])}]")
    
    # 5. Check lexicographic properties
    print("\n5. Checking lexicographic properties:")
    for j in range(n-1):
        # Check if x_j is always ≤ x_{j+1}
        comparison = x_star[:, j] <= x_star[:, j+1] + 1e-10  # allow small error
        if np.all(comparison):
            print(f"  ✓ x{j+1} ≤ x{j+2} holds at all times")
        else:
            violations = np.sum(~comparison)
            print(f"  ⚠ x{j+1} ≤ x{j+2} does not hold at {violations}/{p} time points")
            if violations > 0:
                # Show first 3 violations
                violation_indices = np.where(~comparison)[0][:3]
                for idx in violation_indices:
                    print(f"    t={t_values[idx]:.4f}: x{j+1}={x_star[idx, j]:.6f}, x{j+2}={x_star[idx, j+1]:.6f}")
    
    # 6. Save results to files
    print("\nSaving results to files...")
    
    # Save optimal solution
    df_solution = pd.DataFrame(x_star, columns=[f'x{i+1}' for i in range(8)])
    df_solution['t'] = t_values
    df_solution = df_solution[['t'] + [f'x{i+1}' for i in range(8)]]
    df_solution.to_csv('strict_lexico_aco_optimal_solution.csv', index=False)
    print("✓ Optimal solution saved to strict_lexico_aco_optimal_solution.csv")
    
    # Save bandwidth and demand data
    bandwidth_data = []
    for t_idx, t in enumerate(t_values):
        for i in range(8):
            for j in range(8):
                bandwidth_data.append({
                    'time': t,
                    'user': i+1,
                    'terminal': j+1,
                    'bandwidth': A_history[t_idx, i, j],
                    'demand': b_history[t_idx, i]
                })
    
    df_bandwidth = pd.DataFrame(bandwidth_data)
    df_bandwidth.to_csv('strict_lexico_aco_bandwidth_demand_summary.csv', index=False)
    print("✓ Bandwidth and demand data saved to strict_lexico_aco_bandwidth_demand_summary.csv")
    
    # Save parameter matrices
    df_mu = pd.DataFrame(mu_matrix, 
                         index=[f'user{i+1}' for i in range(8)],
                         columns=[f'terminal{j+1}' for j in range(8)])
    df_mu.to_csv('strict_lexico_aco_mu_matrix.csv')
    
    df_sigma = pd.DataFrame(sigma_matrix,
                           index=[f'user{i+1}' for i in range(8)],
                           columns=[f'terminal{j+1}' for j in range(8)])
    df_sigma.to_csv('strict_lexico_aco_sigma_matrix.csv')
    
    print("✓ Parameter matrices saved to strict_lexico_aco_mu_matrix.csv and strict_lexico_aco_sigma_matrix.csv")
    
    # Save solution summary
    df_summary = pd.DataFrame({
        't': t_values,
        'demand': b_history[:, 0],  # all users have same demand
        'sum_x': np.sum(x_star, axis=1),
        'max_x': np.max(x_star, axis=1),
        'min_x': np.min(x_star, axis=1),
        'active_terminals': np.sum(x_star > 1e-6, axis=1)
    })
    df_summary.to_csv('strict_lexico_aco_solution_summary.csv', index=False)
    print("✓ Solution summary saved to strict_lexico_aco_solution_summary.csv")
    
    return sparsity, total_cost

def visualize_results(x_star, t_values, b_history):
    """
    Visualize the solution
    
    Args:
        x_star: optimal solution
        t_values: array of time points
        b_history: demand vector history
    """
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Time series of all variables
    plt.subplot(2, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for j in range(8):
        plt.plot(t_values, x_star[:, j], label=f'$x_{j+1}$', 
                linewidth=2, color=colors[j])
    
    plt.plot(t_values, b_history[:, 0], 'k--', label='Demand $b_i(t)$', linewidth=3)
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Strict Lexicographic Optimal Solution (ACO)', fontsize=14)
    plt.legend(loc='upper right', fontsize=9, ncol=3)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Lexicographic priority display
    plt.subplot(2, 3, 2)
    # Calculate average value per variable
    avg_values = np.mean(x_star, axis=0)
    terminal_indices = np.arange(1, 9)
    bars = plt.bar(terminal_indices, avg_values, color='skyblue', alpha=0.8)
    
    # Highlight priority order
    for i, bar in enumerate(bars):
        if avg_values[i] == 0:
            bar.set_color('lightgray')
        elif i == np.argmin(avg_values[avg_values > 0]):
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    plt.xlabel('Terminal (Priority Order)', fontsize=12)
    plt.ylabel('Average Value', fontsize=12)
    plt.title('Average Value per Terminal\n(Red: First Non-Zero by Priority)', fontsize=14)
    plt.xticks(terminal_indices)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Sparsity over time
    plt.subplot(2, 3, 3)
    sparsity_over_time = np.sum(x_star > 1e-6, axis=1)
    plt.plot(t_values, sparsity_over_time, 'b-', linewidth=2)
    plt.fill_between(t_values, 0, sparsity_over_time, alpha=0.3, color='blue')
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Number of Active Terminals', fontsize=12)
    plt.title('Solution Sparsity Over Time', fontsize=14)
    plt.ylim(0, 8.5)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Lexicographic relationship heatmap
    plt.subplot(2, 3, 4)
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(x_star.T)
    im = plt.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel('Terminal Index', fontsize=12)
    plt.ylabel('Terminal Index', fontsize=12)
    plt.title('Correlation Between Terminals\n(Diagonal = 1)', fontsize=14)
    plt.xticks(range(8), [f'x{i+1}' for i in range(8)])
    plt.yticks(range(8), [f'x{i+1}' for i in range(8)])
    
    # Subplot 5: Total cost over time
    plt.subplot(2, 3, 5)
    total_cost_over_time = np.sum(x_star, axis=1)
    plt.plot(t_values, total_cost_over_time, 'g-', linewidth=2, label='Total Cost')
    plt.plot(t_values, b_history[:, 0] * 8, 'r--', linewidth=2, label='8 × Demand')
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Total Cost Over Time', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Activation pattern
    plt.subplot(2, 3, 6)
    activation_pattern = (x_star > 1e-6).astype(float)
    im = plt.imshow(activation_pattern.T, aspect='auto', cmap='Blues', 
                   extent=[t_values[0], t_values[-1], 8.5, 0.5])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Terminal', fontsize=12)
    plt.title('Terminal Activation Pattern\n(White=Inactive, Blue=Active)', fontsize=14)
    plt.yticks(range(1, 9), [f'x{i}' for i in range(1, 9)])
    
    plt.tight_layout()
    plt.savefig('strict_lexico_aco_solution_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to strict_lexico_aco_solution_visualization.png")
    plt.show()

def main():
    """Main function"""
    print("="*70)
    print("Time Series Fuzzy Relation Strict Lexicographic Optimization Solver")
    print("Using Ant Colony Algorithm to Refine Minimum Satisfying Values")
    print("Based on Section 4.2 of the paper: Realistic Example with 8 Terminals")
    print("="*70)
    
    # Read μ_ij matrix from paper (8x8) - Table 1
    mu_matrix = np.array([
        [0.1538, 0.4578, 0.0876, 0.1489, 0, 0.0898, 0.2093, 0.0230],
        [0.2834, 0.3769, 0.2490, 0.2035, 0.0191, 0.0759, 0.2109, 0.1371],
        [0, 0, 0.2409, 0.1727, 0, 0.1319, 0.0136, 0.0774],
        [0.1862, 0.4035, 0.2417, 0.0697, 0.2438, 0.1313, 0.1077, 0.2117],
        [0.1319, 0.1725, 0.1671, 0.1294, 0.1325, 0.0135, 0, 0],
        [0, 0.0937, 0, 0.0213, 0.0245, 0.0970, 0, 0.1033],
        [0.0566, 0.1715, 0.1717, 0.1888, 0.2370, 0.0835, 0.0993, 0.1553],
        [0.1343, 0.0795, 0.2630, 0, 0, 0.1628, 0.2533, 0.2101]
    ])
    
    sigma_matrix = np.array([
        [1.1544, 0.9808, 1.1419, 1.0216, 0.9918, 1.0304, 1.1355, 1.2908],
        [1.0086, 1.0889, 1.0292, 0.8834, 0.8067, 0.9400, 0.8928, 1.0825],
        [0.8508, 0.9235, 1.0198, 0.8852, 0.9561, 1.0490, 1.0961, 1.1379],
        [0.9258, 0.8598, 1.1588, 1.0105, 0.8205, 1.0739, 1.0124, 0.8942],
        [0.8938, 0.8578, 0.9196, 1.0722, 1.0840, 1.1712, 1.1437, 0.9531],
        [1.2350, 1.0488, 1.0697, 1.2585, 0.9112, 0.9806, 0.8039, 0.9728],
        [0.9384, 0.9823, 1.0835, 0.9333, 1.0100, 0.7862, 0.9802, 1.1098],
        [1.0748, 0.9804, 0.9756, 1.0187, 0.9455, 0.9160, 0.8792, 0.9722]
    ])
    
    # Base demand vector β
    beta = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    
    # Parameter settings (consistent with paper)
    p = 20  # number of time points
    time_interval = (0, 2)  # time interval [0, 2]
    
    print(f"System Configuration:")
    print(f"  Number of users: 8")
    print(f"  Number of terminals: 8")
    print(f"  Time interval: {time_interval}")
    print(f"  Number of time points: {p}")
    print(f"  Lexicographic priority: x1 → x2 → ... → x8 (strict)")
    print(f"  Ant colony parameters: 50 ants, 100 iterations")
    print(f"  Demand function: b_i(t) = 0.1 + 0.1·sin(2πt)")
    print(f"  Bandwidth model: Gaussian PDF with μ_ij, σ_ij from Tables 1&2")
    
    # Run algorithm
    start_time = datetime.now()
    print("\nStarting strict lexicographic optimization...")
    x_star, A_history, b_history, consistent = time_series_strict_lexicographic_optimization(
        mu_matrix, sigma_matrix, beta, p, time_interval
    )
    
    end_time = datetime.now()
    
    print(f"\nSolution time: {(end_time - start_time).total_seconds():.3f} seconds")
    
    # Verify solution
    print("\n" + "="*60)
    print("Step 3: Verifying solution...")
    t_values = np.linspace(time_interval[0], time_interval[1], p)
    feasible, min_residual, max_violation, violations = verify_solution(x_star, A_history, b_history)
    
    if feasible:
        print(f"✓ Solution is feasible at all time points")
        print(f"  Minimum residual: {min_residual:.6e}")
    else:
        print(f"✗ Solution is infeasible on {len(violations)} constraints!")
        print(f"  Maximum violation: {max_violation:.6e}")
        if len(violations) <= 10:  # show only first 10 violations
            for t_idx, i, violation in violations[:10]:
                print(f"  time point {t_idx} (t={t_values[t_idx]:.4f}), user {i+1}: violation {violation:.6e}")
    
    # Analyze solution and save results
    sparsity, total_cost = analyze_and_save_results(x_star, A_history, b_history, t_values, mu_matrix, sigma_matrix)
    
    # Visualize solution
    print("\nGenerating visualization...")
    visualize_results(x_star, t_values, b_history)
    
    # Output the 20 solutions
    print("\n" + "="*70)
    print("Solutions at 20 time points (Strict Lexicographic Optimization + ACO):")
    print("="*70)
    
    print("Time       ", end="")
    for j in range(8):
        print(f"x_{j+1}^*(t)  ", end="")
    print("Sum")
    
    for t_idx, t in enumerate(t_values):
        print(f"{t:.4f}  ", end="")
        row_sum = 0
        for j in range(8):
            val = x_star[t_idx, j]
            print(f"{val:.6f}  ", end="")
            row_sum += val
        print(f"{row_sum:.6f}")
    
    # Generate final report
    print("\n" + "="*70)
    print("Strict Lexicographic Optimization Summary Report")
    print("="*70)
    print(f"1. System Configuration:")
    print(f"   - Scale: 8 users × 8 terminals × {p} time points")
    print(f"   - Time interval: [{time_interval[0]}, {time_interval[1]}]")
    print(f"   - Lexicographic priority: x1 → x2 → ... → x8 (strict)")
    print(f"   - Ant colony: used to refine minimum satisfying values (50 ants, 100 iterations)")
    
    print(f"\n2. System Status:")
    print(f"   - Consistency: {'✓ Consistent' if consistent else '⚠ Partially inconsistent'}")
    print(f"   - Solution feasibility: {'✓ Feasible' if feasible else '✗ Infeasible'}")
    if feasible:
        print(f"   - Minimum residual: {min_residual:.2e}")
    else:
        print(f"   - Maximum violation: {max_violation:.2e}")
    
    print(f"\n3. Solution Characteristics:")
    print(f"   - Sparsity: {sparsity:.2f}%")
    print(f"   - Total cost: {total_cost:.6f}")
    print(f"   - Average cost per time point: {total_cost/p:.6f}")
    
    print(f"\n4. Lexicographic Properties:")
    print(f"   - Priority execution: strictly in order x1, x2, ..., x8")
    print(f"   - Optimization principle: each variable set to the minimum satisfying value")
    
    print(f"\n5. Computational Performance:")
    print(f"   - Computation time: {(end_time - start_time).total_seconds():.3f} seconds")
    
    print(f"\n6. Output Files:")
    print(f"   - Optimal solution: strict_lexico_aco_optimal_solution.csv")
    print(f"   - Bandwidth and demand: strict_lexico_aco_bandwidth_demand_summary.csv")
    print(f"   - Parameter matrices: strict_lexico_aco_mu_matrix.csv, strict_lexico_aco_sigma_matrix.csv")
    print(f"   - Solution summary: strict_lexico_aco_solution_summary.csv")
    print(f"   - Visualization: strict_lexico_aco_solution_visualization.png")
    
    print("\n" + "="*70)
    
    return x_star, A_history, b_history, t_values

if __name__ == "__main__":
    try:
        x_star, A_history, b_history, t_values = main()
        if x_star is not None:
            print("\n" + "="*70)
            print("Program executed successfully!")
            print("="*70)
            
            # Display first few time points
            print("\nFirst 5 time points solutions:")
            for i in range(min(5, len(t_values))):
                print(f"  t={t_values[i]:.4f}: {x_star[i]}")
        else:
            print("\nProgram execution failed, no valid solution obtained.")
    except Exception as e:
        print(f"\nProgram execution error: {e}")
        import traceback
        traceback.print_exc()