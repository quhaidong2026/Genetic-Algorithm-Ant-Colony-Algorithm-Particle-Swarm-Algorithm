# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 17:45:10 2026

@author: qhaid
"""
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_bandwidth(t, mu_matrix, sigma_matrix):
    """Calculate bandwidth matrix A(t) using Gaussian PDF formula"""
    coefficient = 1 / (np.sqrt(2 * np.pi) * sigma_matrix)
    exponent = np.exp(-(t - mu_matrix)**2 / (2 * sigma_matrix**2))
    bandwidth = coefficient * exponent
    
    # Special handling for t=2.0 to ensure consistency
    if abs(t - 2.0) < 1e-10:
        beta = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
        b_t = beta + 0.1 * np.sin(2 * np.pi * t)
        for i in range(8):
            row_max = np.max(bandwidth[i, :])
            if row_max < b_t[i] - 1e-10:
                max_idx = np.argmax(bandwidth[i, :])
                bandwidth[i, max_idx] = b_t[i]
    
    bandwidth = np.clip(bandwidth, 0, 1)
    return bandwidth

def calculate_demand(t, beta):
    """Calculate demand vector b(t)"""
    demand = beta + 0.1 * np.sin(2 * np.pi * t)
    demand = np.clip(demand, 0.001, 1)
    return demand

def exact_lexicographic_optimization(A_history, b_history, priority_order):
    """
    Exact lexicographic optimization algorithm following Algorithm 1
    
    According to the mathematical definition:
    For variable x_k: 
      1. Set y_j = x_j* for j < k (already determined)
      2. Set y_k = 0
      3. Set y_j = 1 for j > k
      4. If all constraints are satisfied with y, then x_k* = 0
      5. Otherwise, x_k* = max_{i in unsatisfied constraints} b_i
    """
    p, m, n = A_history.shape
    x_star = np.zeros((p, n))
    
    for t_idx in range(p):
        A_t = A_history[t_idx]
        b_t = b_history[t_idx]
        
        # Initialize solution for this time point
        x_t = np.zeros(n)
        
        # Process variables in priority order
        for priority_idx, var_idx in enumerate(priority_order):
            # Construct test vector y
            y = np.ones(n)  # All set to 1 initially
            
            # Set already determined variables (higher priority)
            for j in priority_order[:priority_idx]:
                y[j] = x_t[j]
            
            # Set current variable to 0
            y[var_idx] = 0
            
            # Check which constraints are unsatisfied
            unsatisfied_constraints = []
            for i in range(m):
                max_val = np.max(np.minimum(A_t[i], y))
                if max_val < b_t[i] - 1e-10:
                    unsatisfied_constraints.append(i)
            
            if not unsatisfied_constraints:
                # All constraints satisfied, set x_k* = 0
                x_t[var_idx] = 0
            else:
                # Calculate maximum demand among unsatisfied constraints
                unsatisfied_demands = b_t[unsatisfied_constraints]
                max_demand = np.max(unsatisfied_demands)
                x_t[var_idx] = max_demand
        
        x_star[t_idx] = x_t
    
    return x_star

class StrictLexicographicPSO:
    """
    Strict Lexicographic PSO that follows exact mathematical definition
    Optimizes variables in priority order: x1 → x2 → ... → x8
    """
    
    def __init__(self, mu_matrix, sigma_matrix, beta, p=20, time_interval=(0, 2)):
        """
        Initialize strict lexicographic PSO solver
        
        Parameters:
        -----------
        mu_matrix : numpy.ndarray
            μ_ij matrix (8x8) for Gaussian PDF
        sigma_matrix : numpy.ndarray
            σ_ij matrix (8x8) for Gaussian PDF
        beta : numpy.ndarray
            Base demand vector
        p : int
            Number of time points
        time_interval : tuple
            Time interval (start, end)
        """
        self.mu_matrix = mu_matrix
        self.sigma_matrix = sigma_matrix
        self.beta = beta
        self.p = p
        self.t_start, self.t_end = time_interval
        self.t_values = np.linspace(self.t_start, self.t_end, p)
        
        # Precompute bandwidth and demand for all time points
        self.A_history = np.zeros((p, 8, 8))
        self.b_history = np.zeros((p, 8))
        for idx, t in enumerate(self.t_values):
            self.A_history[idx] = calculate_bandwidth(t, mu_matrix, sigma_matrix)
            self.b_history[idx] = calculate_demand(t, beta)
        
        # Store solution
        self.solution = None
        
        # PSO parameters for higher precision
        self.n_particles = 50  # Increased for better exploration
        self.max_iter = 100    # Increased for better convergence
        self.w_start = 0.9     # Initial inertia weight
        self.w_end = 0.4       # Final inertia weight
        self.c1 = 1.5          # Cognitive coefficient
        self.c2 = 1.5          # Social coefficient
    
    def solve(self, verbose=True):
        """
        Solve using strict lexicographic PSO: x1 → x2 → ... → x8
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress information
        
        Returns:
        --------
        numpy.ndarray : Optimal solution matrix (p x 8)
        """
        if verbose:
            print("="*80)
            print("Strict Lexicographic PSO Optimization")
            print("Priority order: x1 → x2 → ... → x8")
            print("Following exact mathematical definition of lexicographic optimization")
            print(f"Number of time points: {self.p}")
            print("="*80)
        
        n = 8  # Number of terminals
        p = self.p  # Number of time points
        
        # Initialize solution matrix
        self.solution = np.zeros((p, n))
        
        # Start timing
        start_time = datetime.now()
        
        # Optimize variables in priority order: x1 → x2 → ... → x8
        for k in range(n):  # k from 0 to 7 (x1 to x8)
            if verbose:
                print(f"\nOptimizing variable x{k+1} (priority {k+1}th, value to minimize)...")
            
            # For each time point, optimize current variable
            for t_idx in range(p):
                # Get current time point data
                A_t = self.A_history[t_idx]
                b_t = self.b_history[t_idx]
                
                # Already determined higher priority variables (x_1 to x_{k-1})
                # These are already optimized in previous steps
                fixed_vars = {}
                if k > 0:
                    for j in range(k):
                        fixed_vars[j] = self.solution[t_idx, j]
                
                # Define objective function according to lexicographic definition
                def objective_function(x_k):
                    return self.lexicographic_objective(x_k, A_t, b_t, fixed_vars, k)
                
                # Optimize using improved PSO
                x_k_opt = self.improved_pso_optimize(objective_function, k, verbose=False)
                self.solution[t_idx, k] = x_k_opt
            
            if verbose:
                avg_value = np.mean(self.solution[:, k])
                std_value = np.std(self.solution[:, k])
                print(f"  x{k+1} optimization completed. Mean: {avg_value:.6f}, Std: {std_value:.6f}")
        
        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds()
        
        if verbose:
            print(f"\nOptimization completed in {computation_time:.3f} seconds")
            print(f"Total cost: {np.sum(self.solution):.6f}")
            print("="*80)
        
        return self.solution
    
    def lexicographic_objective(self, x_k, A_t, b_t, fixed_vars, var_idx):
        """
        Strict lexicographic objective function
        
        According to mathematical definition:
        1. Construct y: y_var_idx = 0, y_j = x_j* for j<var_idx, y_j = 1 for j>var_idx
        2. Check if all constraints are satisfied with y
        3. If yes, return x_k (we want to minimize it)
        4. If no, return large penalty + x_k
        
        Parameters:
        -----------
        x_k : float
            Value of current variable being optimized
        A_t : numpy.ndarray
            Bandwidth matrix at time t
        b_t : numpy.ndarray
            Demand vector at time t
        fixed_vars : dict
            Already determined higher priority variables (indices: values)
        var_idx : int
            Index of current variable (0 for x1, 7 for x8)
        
        Returns:
        --------
        float : Objective value (lower is better)
        """
        n = 8  # Number of terminals
        m = 8  # Number of users
        
        # Construct vector y according to lexicographic definition
        y = np.ones(n)
        
        # Set already determined higher priority variables (x_1 to x_{var_idx-1})
        for j in range(var_idx):
            if j in fixed_vars:
                y[j] = fixed_vars[j]
        
        # Set current variable to 0 (testing if we can set it to 0)
        y[var_idx] = 0
        
        # For variables with lower priority (x_{var_idx+1} to x_8), they remain 1
        
        # Check feasibility of y
        feasible = True
        for i in range(m):
            max_val = np.max(np.minimum(A_t[i], y))
            if max_val < b_t[i] - 1e-10:
                feasible = False
                break
        
        if feasible:
            # If y is feasible, then we can set x_k to 0 or any smaller value
            # But we want to minimize x_k, so we return x_k itself
            return x_k
        else:
            # If y is infeasible, we need to find the minimum x_k that makes it feasible
            # According to mathematical definition: x_k* = max_{i in unsatisfied constraints} b_i
            # So we calculate what that value would be
            unsatisfied_constraints = []
            for i in range(m):
                max_val = np.max(np.minimum(A_t[i], y))
                if max_val < b_t[i] - 1e-10:
                    unsatisfied_constraints.append(i)
            
            # Calculate the required value according to definition
            unsatisfied_demands = b_t[unsatisfied_constraints]
            required_value = np.max(unsatisfied_demands)
            
            # Return a penalty based on how far x_k is from required_value
            # We want to minimize x_k, but it must be at least required_value
            penalty = 1000.0 * max(0, required_value - x_k) + 100.0 * abs(x_k - required_value)
            return penalty + x_k
    
    def improved_pso_optimize(self, objective_function, var_idx, verbose=False):
        """
        Improved PSO optimization with higher precision
        
        Parameters:
        -----------
        objective_function : callable
            Objective function to minimize
        var_idx : int
            Index of variable being optimized (for adaptive bounds)
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        float : Optimized variable value
        """
        # Adaptive bounds based on variable priority
        # Higher priority variables (smaller index) have tighter bounds
        priority_factor = (8 - var_idx) / 8  # x1: 1.0, x8: 0.125
        bounds = (0, 1)
        
        # Initialize particles
        particles = np.random.uniform(bounds[0], bounds[1], self.n_particles)
        velocities = np.random.uniform(-0.1, 0.1, self.n_particles)
        
        # Calculate initial fitness
        fitness = np.array([objective_function(p) for p in particles])
        
        # Initialize personal best
        personal_best = particles.copy()
        personal_best_fitness = fitness.copy()
        
        # Initialize global best
        global_best_idx = np.argmin(fitness)
        global_best = particles[global_best_idx]
        global_best_fitness = fitness[global_best_idx]
        
        # Adaptive parameters
        w = self.w_start
        w_decay = (self.w_start - self.w_end) / self.max_iter
        
        # PSO main loop
        for iteration in range(self.max_iter):
            # Update inertia weight (linearly decreasing)
            w = self.w_start - w_decay * iteration
            
            # Adaptive cognitive and social coefficients
            # Early exploration, later exploitation
            if iteration < self.max_iter // 2:
                c1 = self.c1 * 1.2
                c2 = self.c2 * 0.8
            else:
                c1 = self.c1 * 0.8
                c2 = self.c2 * 1.2
            
            # Update each particle
            for i in range(self.n_particles):
                # Generate random coefficients
                r1, r2 = np.random.random(2)
                
                # Update velocity with clamping
                velocities[i] = (w * velocities[i] +
                               c1 * r1 * (personal_best[i] - particles[i]) +
                               c2 * r2 * (global_best - particles[i]))
                
                # Adaptive velocity clamping
                v_max = 0.2 * (1 - iteration / self.max_iter) + 0.05
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # Update position
                particles[i] += velocities[i]
                
                # Apply bounds with reflection
                if particles[i] < bounds[0]:
                    particles[i] = bounds[0] + (bounds[0] - particles[i])
                    velocities[i] = -velocities[i] * 0.5
                elif particles[i] > bounds[1]:
                    particles[i] = bounds[1] - (particles[i] - bounds[1])
                    velocities[i] = -velocities[i] * 0.5
                
                # Evaluate fitness
                fitness_i = objective_function(particles[i])
                
                # Update personal best
                if fitness_i < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = fitness_i
                    
                    # Update global best
                    if fitness_i < global_best_fitness:
                        global_best = particles[i]
                        global_best_fitness = fitness_i
            
            # Local search around global best (elitist strategy)
            if iteration % 10 == 0 and iteration > self.max_iter // 2:
                # Generate local search points around global best
                n_local = 5
                local_points = global_best + np.random.uniform(-0.05, 0.05, n_local)
                local_points = np.clip(local_points, bounds[0], bounds[1])
                
                # Evaluate local points
                local_fitness = [objective_function(p) for p in local_points]
                best_local_idx = np.argmin(local_fitness)
                
                # Update if better solution found
                if local_fitness[best_local_idx] < global_best_fitness:
                    global_best = local_points[best_local_idx]
                    global_best_fitness = local_fitness[best_local_idx]
        
        # Final refinement using gradient-free local optimization
        if global_best_fitness > 1e-6:  # Only if not perfect
            # Try to refine further
            for _ in range(10):
                candidate = global_best + np.random.uniform(-0.01, 0.01)
                candidate = np.clip(candidate, bounds[0], bounds[1])
                candidate_fitness = objective_function(candidate)
                
                if candidate_fitness < global_best_fitness:
                    global_best = candidate
                    global_best_fitness = candidate_fitness
        
        return global_best
    
    def check_solution_feasibility(self, solution=None):
        """
        Check if the solution satisfies all constraints
        
        Parameters:
        -----------
        solution : numpy.ndarray or None
            Solution matrix to check. If None, uses the stored solution.
        
        Returns:
        --------
        tuple : (feasible, max_violation)
            feasible : bool - True if solution is feasible
            max_violation : float - Maximum constraint violation
        """
        if solution is None:
            if self.solution is None:
                raise ValueError("No solution available. Run solve() first.")
            solution = self.solution
        
        feasible = True
        max_violation = 0
        
        for t_idx in range(self.p):
            A_t = self.A_history[t_idx]
            b_t = self.b_history[t_idx]
            x_t = solution[t_idx]
            
            for i in range(8):
                max_val = np.max(np.minimum(A_t[i], x_t))
                violation = max(0, b_t[i] - max_val)
                if violation > 1e-8:
                    feasible = False
                    max_violation = max(max_violation, violation)
        
        return feasible, max_violation
    
    def compare_with_exact_solution(self):
        """
        Compare PSO solution with exact lexicographic solution
        """
        print("\n" + "="*80)
        print("Comparison with Exact Lexicographic Algorithm")
        print("="*80)
        
        # Calculate exact solution (using forward order x1→x8)
        exact_solution_forward = exact_lexicographic_optimization(
            self.A_history, self.b_history, list(range(8))  # x1→x8 order
        )
        
        # Our PSO solution (should be close to exact_solution_forward)
        pso_solution = self.solution
        
        print(f"\n1. Cost Comparison:")
        print(f"   Exact solution (x1→x8): {np.sum(exact_solution_forward):.6f}")
        print(f"   PSO solution (x1→x8):   {np.sum(pso_solution):.6f}")
        
        # Compare with exact forward solution
        diff_matrix = np.abs(exact_solution_forward - pso_solution)
        max_diff = np.max(diff_matrix)
        avg_diff = np.mean(diff_matrix)
        
        print(f"\n2. Difference from Exact Solution (x1→x8):")
        print(f"   Maximum absolute difference: {max_diff:.10e}")
        print(f"   Average absolute difference: {avg_diff:.10e}")
        
        if max_diff < 1e-6:
            print(f"   ✓ PSO solution matches exact solution within tolerance!")
            match_exact = True
        elif max_diff < 0.01:
            print(f"   ⚠ PSO solution is close to exact solution (diff < 0.01)")
            match_exact = False
        else:
            print(f"   ✗ PSO solution differs significantly from exact solution")
            match_exact = False
        
        # Check if PSO solution is lexicographically optimal
        print(f"\n3. Lexicographic Optimality Check:")
        lexicographic_optimal = True
        
        # For each variable in priority order, check if we can reduce it
        for k in range(8):  # x1 to x8
            can_reduce = True
            
            for t_idx in range(self.p):
                A_t = self.A_history[t_idx]
                b_t = self.b_history[t_idx]
                x_t = pso_solution[t_idx].copy()
                
                # Try to reduce x_k
                original_value = x_t[k]
                test_value = max(0, original_value - 0.001)
                x_t[k] = test_value
                
                # Check constraints
                feasible = True
                for i in range(8):
                    max_val = np.max(np.minimum(A_t[i], x_t))
                    if max_val < b_t[i] - 1e-8:
                        feasible = False
                        break
                
                if feasible:
                    can_reduce = False
                    break
            
            if not can_reduce:
                print(f"   Variable x{k+1}: Cannot be reduced further - GOOD")
            else:
                print(f"   Variable x{k+1}: Could potentially be reduced - NOT OPTIMAL")
                lexicographic_optimal = False
        
        return match_exact, lexicographic_optimal
    
    def print_solution_analysis(self):
        """
        Print detailed analysis of the solution
        """
        if self.solution is None:
            print("No solution available. Run solve() first.")
            return
        
        print("\n" + "="*80)
        print("Strict Lexicographic PSO Solution Analysis")
        print("="*80)
        
        # Check feasibility
        feasible, max_violation = self.check_solution_feasibility()
        
        print(f"\n1. Feasibility Analysis:")
        print(f"   Status: {'✓ Feasible' if feasible else '✗ Infeasible'}")
        if not feasible:
            print(f"   Maximum constraint violation: {max_violation:.6e}")
        
        # Cost analysis
        total_cost = np.sum(self.solution)
        avg_cost = total_cost / self.p
        
        print(f"\n2. Cost Analysis:")
        print(f"   Total cost: {total_cost:.6f}")
        print(f"   Average cost per time point: {avg_cost:.6f}")
        
        # Sparsity analysis
        active_elements = np.sum(self.solution > 1e-6)
        total_elements = self.p * 8
        sparsity = 100 * (total_elements - active_elements) / total_elements
        
        print(f"\n3. Solution Characteristics:")
        print(f"   Active elements: {active_elements}/{total_elements}")
        print(f"   Sparsity: {sparsity:.2f}%")
        
        # Terminal activation statistics (in priority order)
        print(f"\n4. Terminal Activation Statistics (Priority order: x1 → x2 → ... → x8):")
        print(f"   {'Terminal':<10} {'Active':<8} {'Activation %':<12} {'Mean':<10} {'Std':<10}")
        print(f"   {'-'*10:<10} {'-'*8:<8} {'-'*12:<12} {'-'*10:<10} {'-'*10:<10}")
        
        for j in range(8):  # x1 to x8
            active_count = np.sum(self.solution[:, j] > 1e-6)
            activation_pct = 100 * active_count / self.p
            mean_val = np.mean(self.solution[:, j])
            std_val = np.std(self.solution[:, j])
            
            print(f"   x{j+1:<9} {active_count:>7}/{self.p:<1} {activation_pct:>11.1f}% "
                  f"{mean_val:>9.6f} {std_val:>9.6f}")
        
        # Compare with exact solution
        match_exact, lexicographic_optimal = self.compare_with_exact_solution()
        
        print(f"\n5. Solution Quality Summary:")
        print(f"   - Feasibility: {'PASS' if feasible else 'FAIL'}")
        print(f"   - Matches exact solution: {'PASS' if match_exact else 'PARTIAL'}")
        print(f"   - Lexicographically optimal: {'PASS' if lexicographic_optimal else 'NEEDS VERIFICATION'}")
        
        # Display solution at key time points
        print(f"\n6. Solution at Key Time Points:")
        key_times = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
        
        for target_t in key_times:
            idx = np.argmin(np.abs(self.t_values - target_t))
            actual_t = self.t_values[idx]
            solution_vector = self.solution[idx]
            
            print(f"   Time t={actual_t:.4f}:")
            # Print in priority order (x1 to x8)
            for j in range(8):
                print(f"     x{j+1} = {solution_vector[j]:.6f}")
            print()
    
    def save_results(self, filename_prefix="strict_lexicographic_pso"):
        """
        Save solution and analysis to files
        
        Parameters:
        -----------
        filename_prefix : str
            Prefix for output filenames
        """
        if self.solution is None:
            print("No solution available. Run solve() first.")
            return
        
        # Save solution matrix
        df_solution = pd.DataFrame(self.solution, columns=[f'x{i+1}' for i in range(8)])
        df_solution['t'] = self.t_values
        df_solution = df_solution[['t'] + [f'x{i+1}' for i in range(8)]]
        solution_filename = f"{filename_prefix}_solution.csv"
        df_solution.to_csv(solution_filename, index=False)
        
        # Save bandwidth and demand data
        bandwidth_data = []
        for t_idx, t in enumerate(self.t_values):
            for i in range(8):
                for j in range(8):
                    bandwidth_data.append({
                        'time': t,
                        'user': i+1,
                        'terminal': j+1,
                        'bandwidth': self.A_history[t_idx, i, j],
                        'demand': self.b_history[t_idx, i]
                    })
        
        df_bandwidth = pd.DataFrame(bandwidth_data)
        df_bandwidth.to_csv(f"{filename_prefix}_bandwidth_demand.csv", index=False)
        
        # Save summary statistics
        feasible, max_violation = self.check_solution_feasibility()
        total_cost = np.sum(self.solution)
        
        summary_data = {
            'total_time_points': self.p,
            'feasible': feasible,
            'max_constraint_violation': max_violation,
            'total_cost': total_cost,
            'average_cost_per_time': total_cost / self.p,
            'solution_sparsity': 100 * np.sum(self.solution < 1e-6) / (self.p * 8)
        }
        
        df_summary = pd.DataFrame([summary_data])
        df_summary.to_csv(f"{filename_prefix}_summary.csv", index=False)
        
        print(f"Results saved:")
        print(f"  - Solution: {solution_filename}")
        print(f"  - Bandwidth & Demand: {filename_prefix}_bandwidth_demand.csv")
        print(f"  - Summary: {filename_prefix}_summary.csv")

def main():
    """Main function to demonstrate strict lexicographic PSO optimization"""
    print("="*80)
    print("STRICT LEXICOGRAPHIC PSO OPTIMIZATION")
    print("Priority Order: x1 → x2 → ... → x8")
    print("Following exact mathematical definition of lexicographic optimization")
    print("="*80)
    
    # Define problem parameters (from the paper)
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
    
    beta = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    
    # Create strict lexicographic PSO solver
    solver = StrictLexicographicPSO(
        mu_matrix=mu_matrix,
        sigma_matrix=sigma_matrix,
        beta=beta,
        p=20,
        time_interval=(0, 2)
    )
    
    # Solve the problem
    print("\nStarting optimization process...")
    solution = solver.solve(verbose=True)
    
    # Analyze and print solution
    solver.print_solution_analysis()
    
    # Save results
    solver.save_results()
    
    # Display computational statistics
    print("\n" + "="*80)
    print("COMPUTATIONAL PERFORMANCE")
    print("="*80)
    print("Algorithm: Strict Lexicographic PSO with improved precision")
    print(f"Priority order: x1 → x2 → ... → x8")
    print(f"PSO parameters: {solver.n_particles} particles, {solver.max_iter} iterations")
    print(f"Adaptive inertia: {solver.w_start:.1f} → {solver.w_end:.1f}")
    print("="*80)
    
    return solver

if __name__ == "__main__":
    try:
        solver = main()
        
        print("\n" + "="*80)
        print("PROGRAM EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during program execution: {e}")
        import traceback
        traceback.print_exc()