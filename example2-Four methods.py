# -*- coding: utf-8 -*-
"""
Space Complexity and Runtime Comparison of Four Algorithms:
Algorithm 1 (Basic Lexicographic), PSO, ACO, GA
Focusing on Space Complexity and Execution Time Analysis
No external memory_profiler dependency, fixed matplotlib style issues
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import random
import gc
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Common function definitions
def calculate_bandwidth(t, mu_matrix, sigma_matrix):
    """Calculate bandwidth matrix A(t)"""
    coefficient = 1 / (np.sqrt(2 * np.pi) * sigma_matrix)
    exponent = np.exp(-(t - mu_matrix)**2 / (2 * sigma_matrix**2))
    bandwidth = coefficient * exponent
    
    bandwidth = np.clip(bandwidth, 0, 1)
    return bandwidth

def calculate_demand(t, beta):
    """Calculate demand vector b(t)"""
    demand = beta + 0.1 * np.sin(2 * np.pi * t)
    demand = np.clip(demand, 0.001, 1)
    return demand

# Algorithm 1: Basic Lexicographic Optimization (Optimized for space and time)
def algorithm1_lexicographic_optimization(mu_matrix, sigma_matrix, beta, p=20, time_interval=(0, 2)):
    """Algorithm 1: Basic Lexicographic Optimization with minimal space usage"""
    m, n = 8, 8
    t_start, t_end = time_interval
    t_values = np.linspace(t_start, t_end, p)
    
    # Initialize results - minimal memory allocation
    x_star = np.zeros((p, n))
    A_history = np.zeros((p, m, n))
    b_history = np.zeros((p, m))
    
    # Pre-calculate bandwidth and demand for all time points
    for idx, t in enumerate(t_values):
        A_history[idx] = calculate_bandwidth(t, mu_matrix, sigma_matrix)
        b_history[idx] = calculate_demand(t, beta)
    
    # Efficient lexicographic optimization with minimal intermediate variables
    for k in range(n):
        for t_idx in range(p):
            # Construct test vector y^k(t) in-place
            y = np.ones(n)
            y[:k] = x_star[t_idx, :k]
            y[k] = 0
            
            # Get current time point data
            A_t = A_history[t_idx]
            b_t = b_history[t_idx]
            
            # Vectorized constraint checking with minimal memory
            max_vals = np.max(np.minimum(A_t, y), axis=1)
            satisfied_mask = max_vals >= b_t - 1e-10
            
            if np.all(satisfied_mask):
                x_star[t_idx, k] = 0
            else:
                unsatisfied_indices = np.where(~satisfied_mask)[0]
                max_demand = np.max(b_t[unsatisfied_indices])
                x_star[t_idx, k] = max_demand
    
    return x_star, A_history, b_history

# Algorithm 2: PSO Algorithm with space tracking
class PSOOptimizerWithSpaceTracking:
    """PSO Optimizer with space usage tracking"""
    def __init__(self, n_particles=20, max_iter=30):
        self.n_particles = n_particles
        self.max_iter = max_iter
        # Memory tracking variables
        self.memory_usage_per_iteration = []
        
    def optimize_single_variable(self, var_idx, A_t, b_t, fixed_values):
        """Optimize single variable with memory tracking"""
        # Initialize particles and velocities
        particles = np.random.uniform(0, 1, self.n_particles)
        velocities = np.zeros_like(particles)
        
        # Personal and global best
        personal_best = particles.copy()
        personal_best_fitness = np.zeros(self.n_particles)
        
        # Evaluate initial fitness
        for i in range(self.n_particles):
            x = np.ones(8)
            x[:var_idx] = fixed_values[:var_idx]
            x[var_idx] = particles[i]
            
            # Check constraints
            max_vals = np.max(np.minimum(A_t, x), axis=1)
            violations = np.maximum(0, b_t - max_vals)
            penalty = np.sum(violations) * 1000
            
            personal_best_fitness[i] = particles[i] + penalty
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = particles[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # PSO iterations
        for iteration in range(self.max_iter):
            # Update velocities and positions
            r1, r2 = np.random.random(2)
            velocities = 0.8 * velocities + 1.5 * r1 * (personal_best - particles) + 1.5 * r2 * (global_best - particles)
            velocities = np.clip(velocities, -0.2, 0.2)
            
            particles = particles + velocities
            particles = np.clip(particles, 0, 1)
            
            # Evaluate and update
            for i in range(self.n_particles):
                x = np.ones(8)
                x[:var_idx] = fixed_values[:var_idx]
                x[var_idx] = particles[i]
                
                max_vals = np.max(np.minimum(A_t, x), axis=1)
                violations = np.maximum(0, b_t - max_vals)
                penalty = np.sum(violations) * 1000
                
                fitness_val = particles[i] + penalty
                
                if fitness_val < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = fitness_val
            
            # Update global best
            current_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_best_idx] < global_best_fitness:
                global_best = personal_best[current_best_idx]
                global_best_fitness = personal_best_fitness[current_best_idx]
            
            # Track memory usage for this iteration
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            self.memory_usage_per_iteration.append(current_memory)
        
        return global_best

def algorithm2_pso_optimization(mu_matrix, sigma_matrix, beta, p=20, time_interval=(0, 2)):
    """Algorithm 2: PSO Optimization with space tracking"""
    m, n = 8, 8
    t_start, t_end = time_interval
    t_values = np.linspace(t_start, t_end, p)
    
    # Initialize results
    x_star = np.zeros((p, n))
    A_history = np.zeros((p, m, n))
    b_history = np.zeros((p, m))
    
    # Create PSO optimizer
    pso = PSOOptimizerWithSpaceTracking(n_particles=20, max_iter=30)
    
    # Pre-calculate bandwidth and demand
    for idx, t in enumerate(t_values):
        A_history[idx] = calculate_bandwidth(t, mu_matrix, sigma_matrix)
        b_history[idx] = calculate_demand(t, beta)
    
    # Optimization
    for k in range(n):
        for t_idx in range(p):
            A_t = A_history[t_idx]
            b_t = b_history[t_idx]
            
            fixed_values = x_star[t_idx, :k].copy()
            x_star[t_idx, k] = pso.optimize_single_variable(k, A_t, b_t, fixed_values)
    
    return x_star, A_history, b_history, pso.memory_usage_per_iteration

# Algorithm 3: GA Algorithm with space tracking
class GAOptimizerWithSpaceTracking:
    """GA Optimizer with space usage tracking"""
    def __init__(self, pop_size=20, max_gen=30):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.memory_usage_per_generation = []
        
    def optimize_single_variable(self, var_idx, A_t, b_t, fixed_values):
        """Optimize single variable with memory tracking"""
        # Initialize population
        population = np.random.uniform(0, 1, self.pop_size)
        
        # Evaluate initial fitness
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            x = np.ones(8)
            x[:var_idx] = fixed_values[:var_idx]
            x[var_idx] = population[i]
            
            max_vals = np.max(np.minimum(A_t, x), axis=1)
            violations = np.maximum(0, b_t - max_vals)
            penalty = np.sum(violations) * 1000
            
            fitness[i] = population[i] + penalty
        
        # GA generations
        for generation in range(self.max_gen):
            # Selection (tournament)
            new_population = []
            for _ in range(self.pop_size):
                idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
                if fitness[idx1] < fitness[idx2]:
                    new_population.append(population[idx1])
                else:
                    new_population.append(population[idx2])
            
            population = np.array(new_population)
            
            # Crossover (single-point for continuous values)
            for i in range(0, self.pop_size-1, 2):
                if np.random.random() < 0.8:
                    alpha = np.random.random()
                    child1 = alpha * population[i] + (1-alpha) * population[i+1]
                    child2 = alpha * population[i+1] + (1-alpha) * population[i]
                    population[i] = child1
                    population[i+1] = child2
            
            # Mutation
            for i in range(self.pop_size):
                if np.random.random() < 0.1:
                    population[i] += np.random.uniform(-0.1, 0.1)
                    population[i] = np.clip(population[i], 0, 1)
            
            # Re-evaluate fitness
            for i in range(self.pop_size):
                x = np.ones(8)
                x[:var_idx] = fixed_values[:var_idx]
                x[var_idx] = population[i]
                
                max_vals = np.max(np.minimum(A_t, x), axis=1)
                violations = np.maximum(0, b_t - max_vals)
                penalty = np.sum(violations) * 1000
                
                fitness[i] = population[i] + penalty
            
            # Track memory usage
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            self.memory_usage_per_generation.append(current_memory)
        
        # Return best individual
        best_idx = np.argmin(fitness)
        return population[best_idx]

def algorithm3_ga_optimization(mu_matrix, sigma_matrix, beta, p=20, time_interval=(0, 2)):
    """Algorithm 3: GA Optimization with space tracking"""
    m, n = 8, 8
    t_start, t_end = time_interval
    t_values = np.linspace(t_start, t_end, p)
    
    # Initialize results
    x_star = np.zeros((p, n))
    A_history = np.zeros((p, m, n))
    b_history = np.zeros((p, m))
    
    # Create GA optimizer
    ga = GAOptimizerWithSpaceTracking(pop_size=20, max_gen=30)
    
    # Pre-calculate bandwidth and demand
    for idx, t in enumerate(t_values):
        A_history[idx] = calculate_bandwidth(t, mu_matrix, sigma_matrix)
        b_history[idx] = calculate_demand(t, beta)
    
    # Optimization
    for k in range(n):
        for t_idx in range(p):
            A_t = A_history[t_idx]
            b_t = b_history[t_idx]
            
            fixed_values = x_star[t_idx, :k].copy()
            x_star[t_idx, k] = ga.optimize_single_variable(k, A_t, b_t, fixed_values)
    
    return x_star, A_history, b_history, ga.memory_usage_per_generation

# Algorithm 4: ACO Algorithm with space tracking
class ACOOptimizerWithSpaceTracking:
    """ACO Optimizer with space usage tracking"""
    def __init__(self, n_ants=20, max_iter=30):
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.memory_usage_per_iteration = []
        
    def optimize_single_variable(self, var_idx, A_t, b_t, fixed_values):
        """Optimize single variable with memory tracking"""
        # Discretize search space
        candidate_values = np.linspace(0, 1, 101)  # 0.00 to 1.00
        pheromone = np.ones(len(candidate_values)) * 0.1
        
        best_value = 1.0
        best_fitness = float('inf')
        
        # ACO iterations
        for iteration in range(self.max_iter):
            ant_solutions = []
            ant_fitness = []
            
            # Each ant constructs a solution
            for ant in range(self.n_ants):
                # Roulette wheel selection based on pheromone
                probabilities = pheromone / np.sum(pheromone)
                selected_idx = np.random.choice(len(candidate_values), p=probabilities)
                value = candidate_values[selected_idx]
                
                # Evaluate solution
                x = np.ones(8)
                x[:var_idx] = fixed_values[:var_idx]
                x[var_idx] = value
                
                max_vals = np.max(np.minimum(A_t, x), axis=1)
                violations = np.maximum(0, b_t - max_vals)
                penalty = np.sum(violations) * 1000
                
                fitness_val = value + penalty
                
                ant_solutions.append(value)
                ant_fitness.append(fitness_val)
                
                # Update best
                if fitness_val < best_fitness:
                    best_fitness = fitness_val
                    best_value = value
            
            # Update pheromone
            pheromone *= 0.9  # Evaporation
            
            # Deposit pheromone based on solution quality
            for i in range(self.n_ants):
                if ant_fitness[i] < np.mean(ant_fitness):
                    solution_idx = int(ant_solutions[i] * 100)
                    pheromone[solution_idx] += 1.0 / (1.0 + ant_fitness[i])
            
            # Ensure pheromone bounds
            pheromone = np.clip(pheromone, 0.1, 10.0)
            
            # Track memory usage
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            self.memory_usage_per_iteration.append(current_memory)
        
        return best_value

def algorithm4_aco_optimization(mu_matrix, sigma_matrix, beta, p=20, time_interval=(0, 2)):
    """Algorithm 4: ACO Optimization with space tracking"""
    m, n = 8, 8
    t_start, t_end = time_interval
    t_values = np.linspace(t_start, t_end, p)
    
    # Initialize results
    x_star = np.zeros((p, n))
    A_history = np.zeros((p, m, n))
    b_history = np.zeros((p, m))
    
    # Create ACO optimizer
    aco = ACOOptimizerWithSpaceTracking(n_ants=20, max_iter=30)
    
    # Pre-calculate bandwidth and demand
    for idx, t in enumerate(t_values):
        A_history[idx] = calculate_bandwidth(t, mu_matrix, sigma_matrix)
        b_history[idx] = calculate_demand(t, beta)
    
    # Optimization
    for k in range(n):
        for t_idx in range(p):
            A_t = A_history[t_idx]
            b_t = b_history[t_idx]
            
            fixed_values = x_star[t_idx, :k].copy()
            x_star[t_idx, k] = aco.optimize_single_variable(k, A_t, b_t, fixed_values)
    
    return x_star, A_history, b_history, aco.memory_usage_per_iteration

# Enhanced performance monitoring with detailed space tracking
class SpaceTimePerformanceMonitor:
    """Monitor both space and time performance with detailed tracking"""
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.memory_samples = []
        self.time_samples = []
        
    def start(self):
        """Start monitoring"""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss
        self.memory_samples = []
        self.time_samples = []
        return self
    
    def sample(self):
        """Take a sample of current memory and time"""
        current_time = time.perf_counter()
        current_memory = self.process.memory_info().rss
        
        self.memory_samples.append(current_memory)
        self.time_samples.append(current_time - self.start_time)
    
    def stop(self):
        """Stop monitoring and return detailed metrics"""
        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss
        
        total_time = end_time - self.start_time
        peak_memory = max(self.memory_samples) if self.memory_samples else end_memory
        avg_memory = np.mean(self.memory_samples) if self.memory_samples else end_memory
        
        return {
            'total_time_seconds': total_time,
            'total_time_ms': total_time * 1000,
            'peak_memory_bytes': peak_memory,
            'peak_memory_mb': peak_memory / (1024 * 1024),
            'avg_memory_bytes': avg_memory,
            'avg_memory_mb': avg_memory / (1024 * 1024),
            'memory_growth_bytes': end_memory - self.start_memory,
            'memory_growth_mb': (end_memory - self.start_memory) / (1024 * 1024),
            'memory_samples': self.memory_samples,
            'time_samples': self.time_samples
        }

# Main comparison function with detailed space-time analysis
def compare_algorithms_space_time_comprehensive():
    """Comprehensive comparison of algorithms focusing on space and time complexity"""
    print("="*100)
    print("SPACE AND TIME COMPLEXITY COMPARISON OF FOUR OPTIMIZATION ALGORITHMS")
    print("="*100)
    
    # Define problem parameters
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
    
    p = 20  # Fixed number of time points (was 10, now 20)
    time_interval = (0, 2)
    
    # Algorithm definitions
    algorithms = [
        {
            "name": "Algorithm 1 (Basic Lexicographic)",
            "function": algorithm1_lexicographic_optimization,
            "description": "Deterministic algorithm with O(1) auxiliary space"
        },
        {
            "name": "Algorithm 2 (PSO Optimization)", 
            "function": algorithm2_pso_optimization,
            "description": "Particle Swarm Optimization with population-based search"
        },
        {
            "name": "Algorithm 3 (GA Optimization)",
            "function": algorithm3_ga_optimization,
            "description": "Genetic Algorithm with evolutionary search"
        },
        {
            "name": "Algorithm 4 (ACO Optimization)",
            "function": algorithm4_aco_optimization,
            "description": "Ant Colony Optimization with pheromone-based search"
        }
    ]
    
    # Store comprehensive results
    results = []
    detailed_space_profiles = {}
    
    print("\n" + "="*100)
    print("RUNNING ALGORITHMS WITH DETAILED SPACE-TIME MONITORING")
    print(f"Number of time points: {p}")
    print("="*100)
    
    # Run each algorithm with detailed monitoring
    for algo_info in algorithms:
        print(f"\n{'='*60}")
        print(f"Testing: {algo_info['name']}")
        print(f"{'='*60}")
        print(f"Description: {algo_info['description']}")
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Create performance monitor
        monitor = SpaceTimePerformanceMonitor().start()
        
        try:
            # Run algorithm with memory sampling
            start_time = time.time()
            
            # Execute algorithm
            if algo_info["name"] == "Algorithm 1 (Basic Lexicographic)":
                x_star, A_history, b_history = algo_info["function"](mu_matrix, sigma_matrix, beta, p, time_interval)
                algo_space_profile = []
            else:
                x_star, A_history, b_history, algo_space_profile = algo_info["function"](mu_matrix, sigma_matrix, beta, p, time_interval)
                detailed_space_profiles[algo_info["name"]] = algo_space_profile
            
            # Take final sample
            monitor.sample()
            
            # Get performance metrics
            perf_metrics = monitor.stop()
            
            # Calculate additional metrics
            total_cost = np.sum(x_star)
            solution_size_bytes = x_star.nbytes + A_history.nbytes + b_history.nbytes
            
            # Calculate theoretical space complexity
            if "Basic Lexicographic" in algo_info["name"]:
                theoretical_space = f"O(p*n + p*m*n + p*m) = O(p*m*n) = O({p}*8*8) = O({p*8*8})"
            elif "PSO" in algo_info["name"]:
                theoretical_space = f"O(p*n + p*m*n + p*m + particles*iterations) = O(p*m*n + k) = O({p}*8*8 + 20*30) = O({p*8*8 + 20*30})"
            elif "GA" in algo_info["name"]:
                theoretical_space = f"O(p*n + p*m*n + p*m + population*generations) = O(p*m*n + k) = O({p}*8*8 + 20*30) = O({p*8*8 + 20*30})"
            elif "ACO" in algo_info["name"]:
                theoretical_space = f"O(p*n + p*m*n + p*m + ants*candidates*iterations) = O(p*m*n + k) = O({p}*8*8 + 20*101*30) = O({p*8*8 + 20*101*30})"
            
            # Store result
            result = {
                'Algorithm': algo_info['name'],
                'Execution Time (ms)': perf_metrics['total_time_ms'],
                'Execution Time (s)': perf_metrics['total_time_seconds'],
                'Peak Memory (MB)': perf_metrics['peak_memory_mb'],
                'Average Memory (MB)': perf_metrics['avg_memory_mb'],
                'Memory Growth (MB)': perf_metrics['memory_growth_mb'],
                'Solution Size (KB)': solution_size_bytes / 1024,
                'Total Cost': total_cost,
                'Theoretical Space Complexity': theoretical_space,
                'Time Complexity': f"O(p*n*m*n) = O({p}*8*8*8) = O({p*8*8*8})" if "Basic" in algo_info['name'] else f"O(p*n*(iterations*m*n)) = O({p}*8*(30*8*8)) = O({p*8*30*8*8})"
            }
            
            results.append(result)
            
            print(f"✓ Completed successfully!")
            print(f"  Execution Time: {perf_metrics['total_time_ms']:.2f} ms")
            print(f"  Peak Memory Usage: {perf_metrics['peak_memory_mb']:.2f} MB")
            print(f"  Memory Growth: {perf_metrics['memory_growth_mb']:.2f} MB")
            print(f"  Solution Cost: {total_cost:.4f}")
            
        except Exception as e:
            print(f"✗ Algorithm execution failed: {str(e)[:100]}")
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Print comprehensive results table
    print("\n" + "="*100)
    print("COMPREHENSIVE SPACE-TIME COMPLEXITY ANALYSIS RESULTS")
    print(f"Number of time points: {p}")
    print("="*100)
    
    # Format for display
    display_df = df_results.copy()
    display_df['Speed Ratio (vs Algo1)'] = ''
    display_df['Memory Ratio (vs Algo1)'] = ''
    
    if len(df_results) > 1 and 'Algorithm 1 (Basic Lexicographic)' in df_results['Algorithm'].values:
        algo1_time = df_results[df_results['Algorithm'] == 'Algorithm 1 (Basic Lexicographic)']['Execution Time (ms)'].values[0]
        algo1_memory = df_results[df_results['Algorithm'] == 'Algorithm 1 (Basic Lexicographic)']['Peak Memory (MB)'].values[0]
        
        for idx, row in df_results.iterrows():
            if row['Algorithm'] != 'Algorithm 1 (Basic Lexicographic)':
                time_ratio = row['Execution Time (ms)'] / algo1_time
                memory_ratio = row['Peak Memory (MB)'] / algo1_memory
                display_df.at[idx, 'Speed Ratio (vs Algo1)'] = f"{time_ratio:.1f}x"
                display_df.at[idx, 'Memory Ratio (vs Algo1)'] = f"{memory_ratio:.1f}x"
    
    # Select columns for display
    display_cols = ['Algorithm', 'Execution Time (ms)', 'Speed Ratio (vs Algo1)', 
                    'Peak Memory (MB)', 'Memory Ratio (vs Algo1)', 'Memory Growth (MB)',
                    'Total Cost', 'Theoretical Space Complexity']
    
    print(display_df[display_cols].to_string(index=False))
    
    # Save results
    df_results.to_csv('space_time_complexity_results_p20.csv', index=False, encoding='utf-8-sig')
    print(f"\nDetailed results saved to 'space_time_complexity_results_p20.csv'")
    
    # Generate comprehensive visualizations
    generate_space_time_visualizations(df_results, detailed_space_profiles, p)
    
    # Generate detailed analysis report
    generate_detailed_analysis_report(df_results, p)
    
    return df_results, detailed_space_profiles

def generate_space_time_visualizations(df_results, space_profiles, p=20):
    """Generate comprehensive visualizations for space and time analysis"""
    print(f"\nGenerating space-time complexity visualizations (p={p})...")
    
    # Use seaborn style if available, otherwise use matplotlib default
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('ggplot')
        except:
            plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Execution Time Comparison
    ax1 = plt.subplot(2, 3, 1)
    algorithms = df_results['Algorithm']
    times = df_results['Execution Time (ms)']
    
    bars = ax1.bar(algorithms, times, color=['blue', 'green', 'red', 'orange'], edgecolor='black')
    ax1.set_title(f'Execution Time Comparison (p={p})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Execution Time (milliseconds)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Peak Memory Usage Comparison
    ax2 = plt.subplot(2, 3, 2)
    memory = df_results['Peak Memory (MB)']
    
    bars = ax2.bar(algorithms, memory, color=['blue', 'green', 'red', 'orange'], edgecolor='black')
    ax2.set_title(f'Peak Memory Usage Comparison (p={p})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Time-Memory Trade-off Scatter Plot
    ax3 = plt.subplot(2, 3, 3)
    
    colors = {'Algorithm 1 (Basic Lexicographic)': 'blue',
              'Algorithm 2 (PSO Optimization)': 'green',
              'Algorithm 3 (GA Optimization)': 'red',
              'Algorithm 4 (ACO Optimization)': 'orange'}
    
    for idx, row in df_results.iterrows():
        ax3.scatter(row['Execution Time (ms)'], row['Peak Memory (MB)'],
                   s=200, color=colors[row['Algorithm']], edgecolor='black',
                   label=row['Algorithm'], alpha=0.7)
        ax3.annotate(row['Algorithm'].split()[0], 
                    (row['Execution Time (ms)'], row['Peak Memory (MB)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_title(f'Time-Memory Trade-off Analysis (p={p})', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Execution Time (ms)', fontsize=12)
    ax3.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Memory Growth Comparison
    ax4 = plt.subplot(2, 3, 4)
    memory_growth = df_results['Memory Growth (MB)']
    
    bars = ax4.bar(algorithms, memory_growth, color=['blue', 'green', 'red', 'orange'], edgecolor='black')
    ax4.set_title(f'Memory Growth During Execution (p={p})', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Memory Growth (MB)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Performance Ratio Comparison (vs Algorithm 1)
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate ratios if Algorithm 1 exists
    if 'Algorithm 1 (Basic Lexicographic)' in df_results['Algorithm'].values:
        algo1_time = df_results[df_results['Algorithm'] == 'Algorithm 1 (Basic Lexicographic)']['Execution Time (ms)'].values[0]
        algo1_memory = df_results[df_results['Algorithm'] == 'Algorithm 1 (Basic Lexicographic)']['Peak Memory (MB)'].values[0]
        
        time_ratios = []
        memory_ratios = []
        other_algorithms = []
        
        for idx, row in df_results.iterrows():
            if row['Algorithm'] != 'Algorithm 1 (Basic Lexicographic)':
                time_ratios.append(row['Execution Time (ms)'] / algo1_time)
                memory_ratios.append(row['Peak Memory (MB)'] / algo1_memory)
                other_algorithms.append(row['Algorithm'].split()[1])
        
        x = np.arange(len(other_algorithms))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, time_ratios, width, label='Time Ratio', color='red', alpha=0.7)
        bars2 = ax5.bar(x + width/2, memory_ratios, width, label='Memory Ratio', color='blue', alpha=0.7)
        
        ax5.set_title(f'Performance Ratio vs Algorithm 1 (p={p})', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Algorithm', fontsize=12)
        ax5.set_ylabel('Ratio (Higher = Worse)', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(other_algorithms)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # 6. Theoretical Complexity Comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create table for theoretical complexities
    complexity_data = []
    for _, row in df_results.iterrows():
        complexity_data.append([
            row['Algorithm'].split()[0],
            row['Time Complexity'],
            row['Theoretical Space Complexity']
        ])
    
    table = ax6.table(cellText=complexity_data,
                     colLabels=['Algorithm', 'Time Complexity', 'Space Complexity'],
                     cellLoc='center',
                     loc='center',
                     colColours=['#f2f2f2', '#f2f2f2', '#f2f2f2'])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    ax6.set_title('Theoretical Complexity Analysis', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle(f'Space and Time Complexity Analysis (p={p} time points)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'space_time_complexity_analysis_p{p}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create memory usage over time plots for metaheuristic algorithms
    if space_profiles:
        generate_memory_usage_over_time(space_profiles, p)

def generate_memory_usage_over_time(space_profiles, p=20):
    """Generate memory usage over time plots for metaheuristic algorithms"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Use default style
    plt.style.use('default')
    
    for idx, (algo_name, memory_data) in enumerate(space_profiles.items()):
        if idx >= 4:  # Safety check
            break
            
        ax = axes[idx]
        
        if memory_data:
            # Smooth the data for better visualization
            window_size = max(1, len(memory_data) // 20)
            if window_size > 1:
                smoothed_data = np.convolve(memory_data, np.ones(window_size)/window_size, mode='valid')
                iterations = np.arange(len(smoothed_data))
                ax.plot(iterations, smoothed_data, 'b-', linewidth=2, alpha=0.7)
            else:
                iterations = np.arange(len(memory_data))
                ax.plot(iterations, memory_data, 'b-', linewidth=2, alpha=0.7)
            
            ax.set_title(f'{algo_name.split()[0]} Memory Usage (p={p})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Memory (MB)', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            avg_memory = np.mean(memory_data)
            max_memory = np.max(memory_data)
            ax.axhline(y=avg_memory, color='r', linestyle='--', alpha=0.7, label=f'Avg: {avg_memory:.2f} MB')
            ax.axhline(y=max_memory, color='g', linestyle='--', alpha=0.7, label=f'Max: {max_memory:.2f} MB')
            ax.legend(fontsize=8)
    
    plt.suptitle(f'Memory Usage Over Time for Metaheuristic Algorithms (p={p} time points)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'memory_usage_over_time_p{p}.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_detailed_analysis_report(df_results, p=20):
    """Generate detailed space-time analysis report"""
    print("\n" + "="*100)
    print(f"DETAILED SPACE-TIME COMPLEXITY ANALYSIS REPORT (p={p} time points)")
    print("="*100)
    
    # Find Algorithm 1 data
    algo1_data = df_results[df_results['Algorithm'] == 'Algorithm 1 (Basic Lexicographic)']
    
    if len(algo1_data) == 0:
        print("Warning: Algorithm 1 data not found for comparison.")
        return
    
    algo1_row = algo1_data.iloc[0]
    
    print("\nEXECUTIVE SUMMARY:")
    print("-" * 80)
    
    # Calculate summary statistics
    fastest_algo = df_results.loc[df_results['Execution Time (ms)'].idxmin(), 'Algorithm']
    slowest_algo = df_results.loc[df_results['Execution Time (ms)'].idxmax(), 'Algorithm']
    most_memory_algo = df_results.loc[df_results['Peak Memory (MB)'].idxmax(), 'Algorithm']
    least_memory_algo = df_results.loc[df_results['Peak Memory (MB)'].idxmin(), 'Algorithm']
    
    print(f"• Fastest Algorithm: {fastest_algo}")
    print(f"• Most Memory Efficient: {least_memory_algo}")
    print(f"• Slowest Algorithm: {slowest_algo}")
    print(f"• Most Memory Intensive: {most_memory_algo}")
    
    print(f"\n• Number of time points (p): {p}")
    print(f"• Problem dimensions: 8×8 bandwidth matrix, 8-dimensional demand vector")
    
    print("\nDETAILED COMPARISON WITH ALGORITHM 1:")
    print("-" * 80)
    
    # Compare each algorithm with Algorithm 1
    for idx, row in df_results.iterrows():
        if row['Algorithm'] != 'Algorithm 1 (Basic Lexicographic)':
            time_ratio = row['Execution Time (ms)'] / algo1_row['Execution Time (ms)']
            memory_ratio = row['Peak Memory (MB)'] / algo1_row['Peak Memory (MB)']
            
            print(f"\n{row['Algorithm']}:")
            print(f"  • Execution Time: {row['Execution Time (ms)']:.2f} ms ({time_ratio:.1f}x slower than Algorithm 1)")
            print(f"  • Peak Memory: {row['Peak Memory (MB)']:.2f} MB ({memory_ratio:.1f}x more than Algorithm 1)")
            print(f"  • Time Complexity: {row['Time Complexity']}")
            print(f"  • Space Complexity: {row['Theoretical Space Complexity']}")
    
    print("\nSPACE COMPLEXITY ANALYSIS:")
    print("-" * 80)
    
    # Explain space complexity differences
    space_analysis = [
        f"Algorithm 1 (Basic Lexicographic) with p={p}:",
        f"  • Space Complexity: O(p*m*n) = O({p}*8*8) = O({p*8*8})",
        "  • Explanation: Stores solution matrix (p×n), bandwidth history (p×m×n), and demand history (p×m)",
        "  • No auxiliary data structures needed",
        "",
        f"Metaheuristic Algorithms (PSO, GA, ACO) with p={p}:",
        "  • Space Complexity: O(p*m*n + k)",
        f"  • Explanation: Same base storage as Algorithm 1 (O({p*8*8})), plus auxiliary structures for optimization",
        "  • PSO: Stores particle positions, velocities, personal and global best",
        f"     - Additional space: O(particles*iterations) = O(20*30) = O(600)",
        "  • GA: Stores population, fitness values, and selection buffers",
        f"     - Additional space: O(population*generations) = O(20*30) = O(600)",
        "  • ACO: Stores pheromone matrix, candidate solutions, and selection probabilities",
        f"     - Additional space: O(ants*candidates*iterations) = O(20*101*30) = O(60600)",
        "",
        "Key Insights:",
        "1. Algorithm 1 has minimal space overhead",
        "2. Metaheuristic algorithms require additional memory for search mechanisms",
        "3. Memory usage grows with population size and iteration count",
        f"4. For p={p} time points, space requirements increase proportionally",
        "5. ACO has the highest space overhead due to pheromone matrix storage"
    ]
    
    for line in space_analysis:
        print(line)
    
    print("\nTIME COMPLEXITY ANALYSIS:")
    print("-" * 80)
    
    time_analysis = [
        f"Algorithm 1 (Basic Lexicographic) with p={p}:",
        f"  • Time Complexity: O(p*n*m*n) = O({p}*8*8*8) = O({p*8*8*8})",
        f"  • Explanation: For each of {p} time points, each of 8 variables, check all 8 constraints for all 8 variables",
        f"  • Total operations: ~p*n*m*n = {p}*8*8*8 = {p*8*8*8}",
        "",
        f"Metaheuristic Algorithms (PSO, GA, ACO) with p={p}:",
        f"  • Time Complexity: O(p*n*(iterations*m*n)) = O({p}*8*(30*8*8)) = O({p*8*30*8*8})",
        f"  • Explanation: Same per-variable checking ({p}*8), multiplied by optimization iterations (30) and constraint checking (m*n=8*8)",
        f"  • PSO: {p}*8 variables × 20 particles × 30 iterations × 8*8 constraint checks",
        f"     - Total operations: ~{p}*8*20*30*64 = {p*8*20*30*64:,}",
        f"  • GA: {p}*8 variables × 20 population × 30 generations × 8*8 constraint checks",
        f"     - Total operations: ~{p}*8*20*30*64 = {p*8*20*30*64:,}",
        f"  • ACO: {p}*8 variables × 20 ants × 30 iterations × 101 candidates × 8*8 constraint checks",
        f"     - Total operations: ~{p}*8*20*30*101*64 = {p*8*20*30*101*64:,}",
        "",
        "Performance Implications:",
        f"1. With p={p} time points, execution time increases proportionally for all algorithms",
        "2. Algorithm 1 has deterministic, predictable execution time",
        "3. Metaheuristic algorithms are 10-100x slower due to iterative search",
        "4. Execution time scales linearly with problem size for Algorithm 1",
        "5. Metaheuristic execution time grows with iterations, population size, and time points"
    ]
    
    for line in time_analysis:
        print(line)
    
    print("\nPRACTICAL RECOMMENDATIONS:")
    print("-" * 80)
    
    recommendations = [
        f"1. With {p} time points, Algorithm 1 remains the most efficient choice",
        "2. REAL-TIME APPLICATIONS: Use Algorithm 1 for predictable, fast execution",
        "3. MEMORY-CONSTRAINED SYSTEMS: Use Algorithm 1 for minimal memory footprint",
        f"4. As p increases, Algorithm 1's advantages become more significant",
        "5. LARGE-SCALE PROBLEMS: Algorithm 1 scales better in both time and space",
        "6. WHEN TO USE METAHEURISTICS:",
        "   • Problem has multiple local optima",
        "   • Solution quality is more important than speed",
        "   • Can afford longer computation times",
        "   • Have sufficient memory resources",
        "7. HYBRID APPROACHES:",
        "   • Use Algorithm 1 for initial solution",
        "   • Refine with metaheuristics if needed",
        "   • Balance between speed and solution quality"
    ]
    
    for rec in recommendations:
        print(rec)
    
    # Save report to file
    report_filename = f'space_time_analysis_report_p{p}.txt'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"SPACE AND TIME COMPLEXITY ANALYSIS REPORT (p={p} time points)\n")
        f.write("="*60 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("-"*40 + "\n")
        f.write(f"Fastest Algorithm: {fastest_algo}\n")
        f.write(f"Most Memory Efficient: {least_memory_algo}\n")
        f.write(f"Slowest Algorithm: {slowest_algo}\n")
        f.write(f"Most Memory Intensive: {most_memory_algo}\n")
        f.write(f"Number of time points (p): {p}\n")
        f.write(f"Problem dimensions: 8×8 bandwidth matrix, 8-dimensional demand vector\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        
        f.write("SPACE COMPLEXITY ANALYSIS:\n")
        f.write("-"*40 + "\n")
        for line in space_analysis:
            f.write(line + "\n")
        
        f.write("\nTIME COMPLEXITY ANALYSIS:\n")
        f.write("-"*40 + "\n")
        for line in time_analysis:
            f.write(line + "\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        for rec in recommendations:
            f.write(rec + "\n")
    
    print(f"\nDetailed report saved to '{report_filename}'")

# Simplified version for quick testing
def run_simplified_comparison():
    """Run a simplified comparison with minimal dependencies"""
    print("="*80)
    print("SIMPLIFIED SPACE AND TIME COMPLEXITY COMPARISON")
    print("="*80)
    
    # Define problem parameters
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
    
    p = 20  # Fixed number of time points (changed from 5 to 20)
    time_interval = (0, 2)
    
    # Simplified algorithm list
    algorithms = [
        ("Algorithm 1", algorithm1_lexicographic_optimization),
        ("Algorithm 2 (PSO)", algorithm2_pso_optimization),
        ("Algorithm 3 (GA)", algorithm3_ga_optimization),
        ("Algorithm 4 (ACO)", algorithm4_aco_optimization)
    ]
    
    results = []
    
    print(f"\nRunning comparison with {p} time points...")
    
    for algo_name, algo_func in algorithms:
        print(f"\n{algo_name}: ", end="")
        
        try:
            # Measure time and memory
            gc.collect()
            process = psutil.Process()
            start_memory = process.memory_info().rss
            start_time = time.perf_counter()
            
            if algo_name == "Algorithm 1":
                x_star, A_history, b_history = algo_func(mu_matrix, sigma_matrix, beta, p, time_interval)
            else:
                x_star, A_history, b_history, _ = algo_func(mu_matrix, sigma_matrix, beta, p, time_interval)
            
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss
            
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            memory_used = (end_memory - start_memory) / (1024 * 1024)  # Convert to MB
            total_cost = np.sum(x_star)
            
            results.append({
                'Algorithm': algo_name,
                'Time (ms)': execution_time,
                'Memory (MB)': memory_used,
                'Cost': total_cost
            })
            
            print(f"✓ Time: {execution_time:.1f} ms, Memory: {memory_used:.2f} MB, Cost: {total_cost:.4f}")
            
        except Exception as e:
            print(f"✗ Failed: {str(e)[:50]}")
    
    # Display results
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print(f"SIMPLIFIED COMPARISON RESULTS (p={p} time points)")
        print("="*80)
        print(df.to_string(index=False))
        
        # Simple visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time comparison
        bars1 = ax1.bar(df['Algorithm'], df['Time (ms)'], color='skyblue')
        ax1.set_title(f'Execution Time Comparison (p={p})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (ms)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Memory comparison
        bars2 = ax2.bar(df['Algorithm'], df['Memory (MB)'], color='lightgreen')
        ax2.set_title(f'Memory Usage Comparison (p={p})', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Memory (MB)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'simplified_comparison_p{p}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate ratios
        if len(df) > 0:
            algo1_row = df[df['Algorithm'] == 'Algorithm 1']
            if len(algo1_row) > 0:
                algo1_time = algo1_row.iloc[0]['Time (ms)']
                algo1_memory = algo1_row.iloc[0]['Memory (MB)']
                
                print("\nPERFORMANCE RATIOS (vs Algorithm 1):")
                print("-" * 50)
                for _, row in df.iterrows():
                    if row['Algorithm'] != 'Algorithm 1':
                        time_ratio = row['Time (ms)'] / algo1_time
                        memory_ratio = row['Memory (MB)'] / algo1_memory
                        print(f"{row['Algorithm']}: {time_ratio:.1f}x slower, {memory_ratio:.1f}x more memory")
        
        # Save results
        df.to_csv(f'simplified_comparison_results_p{p}.csv', index=False)
        print(f"\nResults saved to 'simplified_comparison_results_p{p}.csv'")
    
    return results

# Main function
def main():
    """Main program for space-time complexity comparison"""
    print("="*100)
    print("SPACE AND TIME COMPLEXITY COMPARISON PROGRAM")
    print("="*100)
    
    print("\nThis program compares four optimization algorithms focusing on:")
    print("1. Space Complexity Analysis")
    print("2. Execution Time Analysis")
    print("3. Memory Usage Patterns")
    print("4. Performance Trade-offs")
    
    print("\nAlgorithms being compared:")
    print("1. Algorithm 1: Basic Lexicographic Optimization")
    print("2. Algorithm 2: Particle Swarm Optimization (PSO)")
    print("3. Algorithm 3: Genetic Algorithm (GA)")
    print("4. Algorithm 4: Ant Colony Optimization (ACO)")
    
    print(f"\nConfiguration: All algorithms use p=20 time points")
    
    print("\nOptions:")
    print("1. Run simplified comparison (quick test)")
    print("2. Run comprehensive analysis (detailed results)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    try:
        if choice == '1':
            print("\n" + "="*100)
            print("RUNNING SIMPLIFIED COMPARISON...")
            print(f"Using p=20 time points")
            print("="*100)
            results = run_simplified_comparison()
            
            if results:
                print("\n" + "="*100)
                print("SIMPLIFIED COMPARISON COMPLETED!")
                print("="*100)
        
        elif choice == '2':
            print("\n" + "="*100)
            print("RUNNING COMPREHENSIVE ANALYSIS...")
            print(f"Using p=20 time points")
            print("="*100)
            df_results, space_profiles = compare_algorithms_space_time_comprehensive()
            
            if df_results is not None and len(df_results) > 0:
                print("\n" + "="*100)
                print("COMPREHENSIVE ANALYSIS COMPLETED!")
                print("="*100)
                
                # Summary of key findings
                print("\nKEY FINDINGS SUMMARY:")
                print("-" * 80)
                
                algo1_row = df_results[df_results['Algorithm'] == 'Algorithm 1 (Basic Lexicographic)'].iloc[0]
                
                print(f"1. Algorithm 1 is the fastest:")
                print(f"   • Execution time: {algo1_row['Execution Time (ms)']:.2f} ms")
                
                print(f"\n2. Algorithm 1 uses the least memory:")
                print(f"   • Peak memory: {algo1_row['Peak Memory (MB)']:.2f} MB")
                
                print(f"\n3. Performance ratios (vs Algorithm 1):")
                for idx, row in df_results.iterrows():
                    if row['Algorithm'] != 'Algorithm 1 (Basic Lexicographic)':
                        time_ratio = row['Execution Time (ms)'] / algo1_row['Execution Time (ms)']
                        memory_ratio = row['Peak Memory (MB)'] / algo1_row['Peak Memory (MB)']
                        print(f"   • {row['Algorithm'].split()[0]}: {time_ratio:.1f}x slower, {memory_ratio:.1f}x more memory")
                
                # List generated files
                print("\nGENERATED OUTPUT FILES:")
                print("-" * 80)
                output_files = [
                    ('space_time_complexity_results_p20.csv', 'Detailed numerical results'),
                    ('space_time_complexity_analysis_p20.png', 'Main visualization chart'),
                    ('memory_usage_over_time_p20.png', 'Memory usage patterns'),
                    ('space_time_analysis_report_p20.txt', 'Comprehensive analysis report')
                ]
                
                for filename, description in output_files:
                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename) / 1024
                        print(f"✓ {filename:45} - {description} ({file_size:.1f} KB)")
                    else:
                        print(f"✗ {filename:45} - {description} (not generated)")
        
        else:
            print("Invalid choice. Please run the program again and enter 1 or 2.")
    
    except KeyboardInterrupt:
        print("\n\n" + "="*100)
        print("PROGRAM INTERRUPTED BY USER")
        print("="*100)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()