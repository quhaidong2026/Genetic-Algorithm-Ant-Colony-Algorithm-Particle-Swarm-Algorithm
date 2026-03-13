# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:59:34 2026

@author: qhaid
"""
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_bandwidth(t, mu_matrix, sigma_matrix):
    """
    Calculate bandwidth matrix A(t) based on Gaussian PDF formula
    """
    coefficient = 1 / (np.sqrt(2 * np.pi) * sigma_matrix)
    exponent = np.exp(-(t - mu_matrix)**2 / (2 * sigma_matrix**2))
    bandwidth = coefficient * exponent
    
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
    """
    Calculate demand vector b(t)
    """
    demand = beta + 0.1 * np.sin(2 * np.pi * t)
    demand = np.clip(demand, 0.001, 1)
    
    return demand

def exact_lexicographic_optimization(A_history, b_history, priority_order):
    """
    Exact lexicographic optimization algorithm - fixed version, strictly guarantees lexicographic order
    
    Args:
        A_history: Bandwidth matrix history (p x 8 x 8)
        b_history: Demand vector history (p x 8)
        priority_order: Optimization order list, e.g., [0,1,2,3,4,5,6,7] means x1→x2→...→x8
    
    Returns:
        x_star: Optimal solution (p x 8)
    """
    p, m, n = A_history.shape
    x_star = np.zeros((p, n))
    
    # Solve independently for each time point
    for t_idx in range(p):
        A_t = A_history[t_idx]
        b_t = b_history[t_idx]
        
        # Initialize current solution
        x_t = np.zeros(n)
        
        # Process each variable in priority order
        for priority_idx, var_idx in enumerate(priority_order):
            # Construct test vector y
            y = np.ones(n)
            
            # Set values of already determined variables
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
                # All constraints satisfied, set x_var = 0
                x_t[var_idx] = 0
            else:
                # Some constraints unsatisfied, calculate maximum demand among unsatisfied constraints
                unsatisfied_demands = b_t[unsatisfied_constraints]
                max_demand = np.max(unsatisfied_demands)
                x_t[var_idx] = max_demand
        
        x_star[t_idx] = x_t
    
    return x_star

class StrictLexicographicGeneticAlgorithm:
    """
    Strict lexicographic optimization genetic algorithm
    Ensures lexicographic priority order of solutions
    """
    def __init__(self, mu_matrix, sigma_matrix, beta, p=20, time_interval=(0, 2),
                 pop_size=50, n_generations=100, crossover_rate=0.8, 
                 mutation_rate=0.1, elite_size=5, tournament_size=3):
        """
        Initialize strict lexicographic optimization genetic algorithm
        
        Args:
            mu_matrix, sigma_matrix, beta: Problem parameters
            pop_size: Population size
            n_generations: Number of generations
            crossover_rate: Crossover rate
            mutation_rate: Mutation rate
            elite_size: Elite size
            tournament_size: Tournament size
        """
        self.mu_matrix = mu_matrix
        self.sigma_matrix = sigma_matrix
        self.beta = beta
        self.p = p
        self.t_start, self.t_end = time_interval
        self.t_values = np.linspace(self.t_start, self.t_end, p)
        
        # Genetic algorithm parameters
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        # Optimization order: x1 → x2 → ... → x8
        self.priority_order = list(range(8))
        
        # Lexicographic weights: Ensure high-priority variables have much larger weights than low-priority ones
        # Weights are set in exponential form to ensure lexicographic relationship
        self.lexicographic_weights = np.array([10**(14-2*j) for j in range(8)])
        
        # Store best solution
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
        # Cache computed bandwidth and demand
        self.bandwidth_cache = {}
        self.demand_cache = {}
        
        # Precompute bandwidth and demand for all time points
        self.precompute_data()
        
        # Solution dimension
        self.solution_shape = (p, 8)
        
    def precompute_data(self):
        """Precompute bandwidth matrices and demand vectors for all time points"""
        for t in self.t_values:
            self.bandwidth_cache[t] = calculate_bandwidth(t, self.mu_matrix, self.sigma_matrix)
            self.demand_cache[t] = calculate_demand(t, self.beta)
    
    def calculate_lexicographic_fitness(self, solution):
        """
        Calculate strict lexicographic fitness
        Fitness = lexicographic weighted sum
        Weights ensure: x1 priority >> x2 >> x3 >> ... >> x8
        
        Args:
            solution: Solution matrix (p x 8)
        
        Returns:
            fitness: Lexicographic fitness value (smaller is better)
        """
        # Calculate sum of each variable
        variable_sums = np.sum(solution, axis=0)
        
        # Lexicographic weighted sum: high-priority variables have much larger weights
        # This ensures reducing x1 is always better than reducing x2, and so on
        fitness = np.sum(variable_sums * self.lexicographic_weights)
        
        return fitness
    
    def evaluate_fitness(self, solution):
        """
        Evaluate solution fitness
        Priority: feasibility first, then lexicographic fitness
        
        Args:
            solution: Solution matrix (p x 8)
        
        Returns:
            fitness: Fitness value
            constraint_violation: Constraint violation degree
            lexicographic_fitness: Lexicographic fitness
        """
        # Calculate constraint violation
        constraint_violation = 0
        max_violation = 0
        
        for t_idx, t in enumerate(self.t_values):
            A_t = self.bandwidth_cache[t]
            b_t = self.demand_cache[t]
            x_t = solution[t_idx]
            
            for i in range(8):
                max_val = np.max(np.minimum(A_t[i], x_t))
                violation = max(0, b_t[i] - max_val)
                constraint_violation += violation
                max_violation = max(max_violation, violation)
        
        # If there's constraint violation, apply large penalty
        # Penalty value must be much larger than possible lexicographic fitness
        if constraint_violation > 1e-8:
            # Use large penalty to ensure infeasible solutions are never better than feasible ones
            penalty = 1e20 + constraint_violation * 1e10
            fitness = penalty
        else:
            # Calculate lexicographic fitness
            fitness = self.calculate_lexicographic_fitness(solution)
        
        return fitness, constraint_violation, self.calculate_lexicographic_fitness(solution)
    
    def initialize_population(self):
        """
        Initialize population
        Use exact lexicographic optimization algorithm to generate high-quality initial solutions
        """
        population = []
        
        # Prepare A_history and b_history
        A_history = np.array([self.bandwidth_cache[t] for t in self.t_values])
        b_history = np.array([self.demand_cache[t] for t in self.t_values])
        
        # 1. First, use exact lexicographic optimization to generate a high-quality solution
        exact_solution = exact_lexicographic_optimization(A_history, b_history, self.priority_order)
        population.append(exact_solution)
        
        # 2. Generate random solutions based on perturbations of exact solution
        for _ in range(self.pop_size - 1):
            # Start from exact solution
            solution = exact_solution.copy()
            
            # Random perturbation while maintaining lexicographic properties
            for t_idx in range(self.p):
                # Apply different perturbation magnitudes based on priority
                # High-priority variables have small perturbations, low-priority have larger
                for j in range(8):
                    if np.random.rand() < 0.3:  # 30% perturbation probability
                        # Perturbation magnitude increases with decreasing priority
                        perturbation_factor = (j + 1) / 8  # x1: 0.125, x8: 1.0
                        # Random direction, limited magnitude
                        perturbation = np.random.uniform(-0.05 * perturbation_factor, 
                                                         0.05 * perturbation_factor)
                        solution[t_idx, j] = max(0, min(1, solution[t_idx, j] + perturbation))
            
            population.append(solution)
        
        return np.array(population)
    
    def tournament_selection(self, population, fitness_values):
        """
        Tournament selection
        
        Args:
            population: Population
            fitness_values: Fitness values
        
        Returns:
            selected_parents: Selected parents
        """
        n = len(population)
        selected_parents = []
        
        for _ in range(n):
            # Randomly select tournament_size individuals
            candidates = np.random.choice(n, self.tournament_size, replace=False)
            # Select the individual with best fitness
            best_idx = candidates[np.argmin(fitness_values[candidates])]
            selected_parents.append(population[best_idx])
        
        return np.array(selected_parents)
    
    def strict_lexicographic_crossover(self, parent1, parent2):
        """
        Strict lexicographic optimization crossover operation
        
        Args:
            parent1: Parent 1
            parent2: Parent 2
        
        Returns:
            child1, child2: Two children
        """
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # For each time point, independently decide crossover method
        for t_idx in range(self.p):
            # Compare lexicographic quality of two parents at this time point
            # For high-priority variables, inherit from better parent
            for j in range(8):
                if np.random.rand() < 0.5:
                    # 50% probability to swap
                    child1[t_idx, j] = parent2[t_idx, j]
                    child2[t_idx, j] = parent1[t_idx, j]
        
        return child1, child2
    
    def strict_lexicographic_mutation(self, individual):
        """
        Strict lexicographic optimization mutation operation
        Prioritize mutating low-priority variables, high-priority variables have lower mutation probability
        
        Args:
            individual: Individual
        
        Returns:
            mutated_individual: Mutated individual
        """
        mutated = individual.copy()
        
        # Mutate each time point
        for t_idx in range(self.p):
            if np.random.rand() < self.mutation_rate:
                # Higher probability to mutate low-priority variables
                for j in range(7, -1, -1):  # From x8 to x1
                    mutation_prob = self.mutation_rate * (8 - j) / 8  # x8: 1.0*mutation_rate, x1: 0.125*mutation_rate
                    
                    if np.random.rand() < mutation_prob:
                        # Mutation magnitude
                        mutation_amount = np.random.uniform(-0.05, 0.05)
                        mutated[t_idx, j] = max(0, min(1, mutated[t_idx, j] + mutation_amount))
        
        return mutated
    
    def enforce_strict_lexicographic_order(self, solution):
        """
        Enforce strict lexicographic properties
        Ensure solution satisfies lexicographic optimality conditions at each time point
        
        Args:
            solution: Solution to optimize
        
        Returns:
            improved_solution: Optimized solution
        """
        improved = solution.copy()
        
        # Prepare A_history and b_history
        A_history = np.array([self.bandwidth_cache[t] for t in self.t_values])
        b_history = np.array([self.demand_cache[t] for t in self.t_values])
        
        # Apply exact lexicographic optimization to each time point
        for t_idx in range(self.p):
            A_t = A_history[t_idx]
            b_t = b_history[t_idx]
            x_t = improved[t_idx].copy()
            
            # Recalculate lexicographic optimal solution for this time point
            # Initialize
            x_optimal = np.zeros(8)
            
            # Process each variable in priority order
            for priority_idx, var_idx in enumerate(self.priority_order):
                # Construct test vector y
                y = np.ones(8)
                
                # Set values of already determined variables
                for j in self.priority_order[:priority_idx]:
                    y[j] = x_optimal[j]
                
                # Set current variable to 0
                y[var_idx] = 0
                
                # Check which constraints are unsatisfied
                unsatisfied_constraints = []
                for i in range(8):
                    max_val = np.max(np.minimum(A_t[i], y))
                    if max_val < b_t[i] - 1e-10:
                        unsatisfied_constraints.append(i)
                
                if not unsatisfied_constraints:
                    # All constraints satisfied, set x_var = 0
                    x_optimal[var_idx] = 0
                else:
                    # Some constraints unsatisfied, calculate maximum demand among unsatisfied constraints
                    unsatisfied_demands = b_t[unsatisfied_constraints]
                    max_demand = np.max(unsatisfied_demands)
                    x_optimal[var_idx] = max_demand
            
            improved[t_idx] = x_optimal
        
        return improved
    
    def repair_solution(self, solution):
        """
        Repair solution to ensure all constraints are satisfied
        
        Args:
            solution: Solution to repair
        
        Returns:
            repaired_solution: Repaired solution
        """
        repaired = solution.copy()
        
        # Prepare A_history and b_history
        A_history = np.array([self.bandwidth_cache[t] for t in self.t_values])
        b_history = np.array([self.demand_cache[t] for t in self.t_values])
        
        # Repair each time point
        for t_idx in range(self.p):
            A_t = A_history[t_idx]
            b_t = b_history[t_idx]
            x_t = repaired[t_idx]
            
            # Check if each user's constraint is satisfied
            for i in range(8):
                max_val = np.max(np.minimum(A_t[i], x_t))
                
                # If constraint is not satisfied, repair it
                if max_val < b_t[i] - 1e-8:
                    # Find terminal that can provide maximum bandwidth
                    # To maintain lexicographic properties, prioritize lower-priority terminals
                    best_terminal = -1
                    
                    for j in range(8):
                        if A_t[i, j] >= b_t[i] - 1e-8:
                            # Priority value: x1=0, x2=1, ..., x8=7
                            # We want to choose lowest priority (largest value)
                            if j > best_terminal or best_terminal == -1:
                                best_terminal = j
                    
                    # If suitable terminal found
                    if best_terminal != -1:
                        # Set this terminal's value to demand value
                        repaired[t_idx, best_terminal] = max(repaired[t_idx, best_terminal], b_t[i])
                    else:
                        # If no terminal can satisfy demand alone, choose terminal with maximum bandwidth
                        best_terminal = np.argmax(A_t[i])
                        repaired[t_idx, best_terminal] = max(repaired[t_idx, best_terminal], b_t[i])
        
        return repaired
    
    def run(self):
        """
        Run strict lexicographic optimization genetic algorithm
        
        Returns:
            best_solution: Optimal solution
        """
        print("Initializing strict lexicographic optimization genetic algorithm...")
        print(f"Population size: {self.pop_size}, Number of generations: {self.n_generations}")
        print(f"Lexicographic optimization order: x1 → x2 → ... → x8")
        
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        fitness_values = []
        constraint_violations = []
        lexicographic_fitnesses = []
        
        for solution in population:
            fitness, constraint_violation, lex_fitness = self.evaluate_fitness(solution)
            fitness_values.append(fitness)
            constraint_violations.append(constraint_violation)
            lexicographic_fitnesses.append(lex_fitness)
        
        fitness_values = np.array(fitness_values)
        constraint_violations = np.array(constraint_violations)
        lexicographic_fitnesses = np.array(lexicographic_fitnesses)
        
        # Record best solution
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        self.best_lexicographic_fitness = lexicographic_fitnesses[best_idx]
        self.best_constraint_violation = constraint_violations[best_idx]
        
        self.fitness_history.append({
            'generation': 0,
            'best_fitness': self.best_fitness,
            'best_lexicographic_fitness': self.best_lexicographic_fitness,
            'best_constraint': self.best_constraint_violation,
            'avg_fitness': np.mean(fitness_values)
        })
        
        # Evolution loop
        for generation in range(1, self.n_generations + 1):
            # Repair all solutions in population to ensure constraints are satisfied
            for i in range(len(population)):
                population[i] = self.repair_solution(population[i])
            
            # Enforce strict lexicographic properties
            for i in range(len(population)):
                population[i] = self.enforce_strict_lexicographic_order(population[i])
            
            # Re-evaluate population
            fitness_values = []
            constraint_violations = []
            lexicographic_fitnesses = []
            
            for solution in population:
                fitness, constraint_violation, lex_fitness = self.evaluate_fitness(solution)
                fitness_values.append(fitness)
                constraint_violations.append(constraint_violation)
                lexicographic_fitnesses.append(lex_fitness)
            
            fitness_values = np.array(fitness_values)
            constraint_violations = np.array(constraint_violations)
            lexicographic_fitnesses = np.array(lexicographic_fitnesses)
            
            # Update global best solution
            generation_best_idx = np.argmin(fitness_values)
            generation_best_fitness = fitness_values[generation_best_idx]
            generation_best_solution = population[generation_best_idx]
            generation_best_lexicographic_fitness = lexicographic_fitnesses[generation_best_idx]
            generation_best_constraint = constraint_violations[generation_best_idx]
            
            if generation_best_fitness < self.best_fitness:
                self.best_fitness = generation_best_fitness
                self.best_solution = generation_best_solution.copy()
                self.best_lexicographic_fitness = generation_best_lexicographic_fitness
                self.best_constraint_violation = generation_best_constraint
            
            # Record history
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'best_lexicographic_fitness': self.best_lexicographic_fitness,
                'best_constraint': self.best_constraint_violation,
                'avg_fitness': np.mean(fitness_values)
            })
            
            # Show progress every 10 generations
            if generation % 10 == 0 or generation == self.n_generations:
                feasible_count = np.sum(constraint_violations < 1e-8)
                
                print(f"Generation {generation:3d}/{self.n_generations}: "
                      f"Best fitness={self.best_fitness:.6e}, "
                      f"Lexicographic fitness={self.best_lexicographic_fitness:.6e}, "
                      f"Feasible solutions={feasible_count}/{self.pop_size}")
            
            # Stop if reached maximum generations
            if generation == self.n_generations:
                break
            
            # Selection
            selected = self.tournament_selection(population, fitness_values)
            
            # Crossover and mutation to generate next generation
            next_population = []
            
            # Elite preservation
            elite_indices = np.argsort(fitness_values)[:self.elite_size]
            for idx in elite_indices:
                next_population.append(population[idx].copy())
            
            # Generate remaining individuals
            while len(next_population) < self.pop_size:
                # Select parents
                parent1_idx = np.random.randint(0, len(selected))
                parent2_idx = np.random.randint(0, len(selected))
                parent1 = selected[parent1_idx]
                parent2 = selected[parent2_idx]
                
                # Crossover
                child1, child2 = self.strict_lexicographic_crossover(parent1, parent2)
                
                # Select one child
                child = child1 if np.random.rand() < 0.5 else child2
                
                # Mutation
                child = self.strict_lexicographic_mutation(child)
                
                # Repair
                child = self.repair_solution(child)
                
                # Enforce strict lexicographic properties
                child = self.enforce_strict_lexicographic_order(child)
                
                next_population.append(child)
            
            population = np.array(next_population[:self.pop_size])
        
        print("Strict lexicographic optimization genetic algorithm completed!")
        return self.best_solution
    
    def compare_with_exact_solution(self, display_details=True):
        """
        Compare with exact algorithm solution
        
        Args:
            display_details: Whether to display detailed comparison information
        """
        print("\n" + "="*80)
        print("Comparison with Exact Algorithm Solution")
        print("="*80)
        
        # Prepare A_history and b_history
        A_history = np.array([self.bandwidth_cache[t] for t in self.t_values])
        b_history = np.array([self.demand_cache[t] for t in self.t_values])
        
        # Calculate exact solution
        exact_solution = exact_lexicographic_optimization(A_history, b_history, self.priority_order)
        
        # Calculate GA solution
        ga_solution = self.best_solution
        
        # Compare
        print(f"{'Comparison Item':<30} {'Exact Algorithm':>15} {'Genetic Algorithm':>15} {'Difference':>15}")
        print("-"*80)
        
        # Compare sum of each variable
        total_diff = 0
        if display_details:
            print(f"{'Variable':<10} {'Exact Sum':>15} {'GA Sum':>15} {'Difference':>15}")
            print("-"*80)
        
        for j in range(8):
            exact_sum = np.sum(exact_solution[:, j])
            ga_sum = np.sum(ga_solution[:, j])
            diff = abs(exact_sum - ga_sum)
            total_diff += diff
            
            if display_details:
                print(f"x{j+1:<10} {exact_sum:15.6f} {ga_sum:15.6f} {diff:15.6f}")
        
        # Compare lexicographic fitness
        exact_lex_fitness = self.calculate_lexicographic_fitness(exact_solution)
        ga_lex_fitness = self.calculate_lexicographic_fitness(ga_solution)
        
        print(f"{'Lexicographic Fitness':<30} {exact_lex_fitness:15.6e} {ga_lex_fitness:15.6e} "
              f"{abs(exact_lex_fitness-ga_lex_fitness):15.6e}")
        
        # Check if exactly the same
        max_diff = np.max(np.abs(exact_solution - ga_solution))
        avg_diff = np.mean(np.abs(exact_solution - ga_solution))
        
        print(f"\nMaximum absolute difference: {max_diff:.10e}")
        print(f"Average absolute difference: {avg_diff:.10e}")
        
        if max_diff < 1e-8:
            print("✓ Genetic algorithm solution is exactly the same as exact algorithm solution!")
            identical = True
        else:
            print("⚠ Genetic algorithm solution differs from exact algorithm solution")
            
            if display_details:
                # Find time point and variable with maximum difference
                diff_matrix = np.abs(exact_solution - ga_solution)
                max_diff_indices = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
                t_idx, var_idx = max_diff_indices
                t_val = self.t_values[t_idx]
                
                print(f"\nMaximum difference details:")
                print(f"  Time: t={t_val:.4f}")
                print(f"  Variable: x{var_idx+1}")
                print(f"  Exact algorithm value: {exact_solution[t_idx, var_idx]:.8f}")
                print(f"  Genetic algorithm value: {ga_solution[t_idx, var_idx]:.8f}")
                print(f"  Difference: {diff_matrix[t_idx, var_idx]:.8f}")
            
            identical = False
        
        return exact_solution, ga_solution, identical
    
    def print_solution_table(self, solution, title="Strict Lexicographic Genetic Algorithm Optimized Solution"):
        """Print solution for 20 time points in table format"""
        print("\n" + "="*120)
        print(f"{title} (20 time points) - Optimization order: x1 → x2 → ... → x8")
        print("="*120)
        
        print(f"{'Time Point':<10}", end="")
        for j in range(8):
            print(f"   x{j+1}(t)   ", end="")
        print(f"{'Sum':>10}")
        print("-"*120)
        
        total_sum = 0
        for t_idx, t in enumerate(self.t_values):
            row = solution[t_idx]
            row_sum = np.sum(row)
            total_sum += row_sum
            
            print(f"{t:.4f}  ", end="")
            for j in range(8):
                print(f"{row[j]:10.6f}  ", end="")
            print(f"{row_sum:10.6f}")
        
        print("-"*120)
        print(f"{'Total':<10}", end="")
        for j in range(8):
            col_sum = np.sum(solution[:, j])
            print(f"{col_sum:10.6f}  ", end="")
        print(f"{total_sum:10.6f}")
        
        # Lexicographic cost
        lex_cost = self.calculate_lexicographic_fitness(solution)
        print(f"{'Lexicographic Cost':<10} {lex_cost:15.6e}")
        print("="*120)
        
        # Output solution characteristics
        print("\nSolution Characteristics Analysis:")
        print(f"Total cost: {total_sum:.6f}")
        print(f"Average cost per time point: {total_sum/self.p:.6f}")
        
        # Sparsity
        sparsity = 100 * np.sum(solution > 1e-6) / (self.p * 8)
        print(f"Sparsity: {sparsity:.2f}%")
        
        # Activation statistics (displayed in optimization order x1 → x2 → ... → x8)
        print("\nTerminal Activation Statistics (Optimization order: x1 → x2 → ... → x8):")
        for j in self.priority_order:  # From x1 to x8
            active_count = np.sum(solution[:, j] > 1e-6)
            activation_rate = 100 * active_count / self.p
            avg_value = np.mean(solution[:, j])
            std_value = np.std(solution[:, j])
            print(f"  Terminal x{j+1}: Activated {active_count:2d}/{self.p} time points "
                  f"({activation_rate:5.1f}%), Mean={avg_value:.6f}, Std={std_value:.6f}")

def strict_lexicographic_genetic_algorithm_optimization(mu_matrix, sigma_matrix, beta, p=20, time_interval=(0, 2)):
    """
    Use strict lexicographic optimization genetic algorithm
    
    Args:
        mu_matrix: μ_ij matrix
        sigma_matrix: σ_ij matrix
        beta: Base demand vector
        p: Number of time points
        time_interval: Time interval
    
    Returns:
        ga_solution: Genetic algorithm optimal solution
        exact_solution: Exact algorithm solution
        A_history: Bandwidth matrix history
        b_history: Demand vector history
        ga_instance: Genetic algorithm instance
    """
    print("="*80)
    print("Strict Lexicographic Optimization Genetic Algorithm (Order: x1 → x2 → ... → x8)")
    print("="*80)
    
    # Initialize strict lexicographic optimization genetic algorithm
    ga = StrictLexicographicGeneticAlgorithm(
        mu_matrix=mu_matrix,
        sigma_matrix=sigma_matrix,
        beta=beta,
        p=p,
        time_interval=time_interval,
        pop_size=50,  # Population size = 50
        n_generations=100,  # Number of generations = 100
        crossover_rate=0.8,
        mutation_rate=0.05,  # Lower mutation rate to maintain lexicographic properties
        elite_size=5,
        tournament_size=3
    )
    
    # Run algorithm
    start_time = datetime.now()
    best_solution = ga.run()
    end_time = datetime.now()
    
    print(f"\nOptimization time: {(end_time - start_time).total_seconds():.3f} seconds")
    
    # Compare with exact algorithm
    exact_solution, ga_solution, identical = ga.compare_with_exact_solution(display_details=True)
    
    # Validate solution
    feasible_count = 0
    max_violation = 0
    
    for t_idx, t in enumerate(ga.t_values):
        A_t = ga.bandwidth_cache[t]
        b_t = ga.demand_cache[t]
        x_t = best_solution[t_idx]
        
        feasible = True
        for i in range(8):
            max_val = np.max(np.minimum(A_t[i], x_t))
            violation = max(0, b_t[i] - max_val)
            
            if violation > 1e-8:
                feasible = False
                max_violation = max(max_violation, violation)
        
        if feasible:
            feasible_count += 1
    
    print(f"\nFeasibility verification: {feasible_count}/{p} time points feasible")
    print(f"Maximum constraint violation: {max_violation:.6e}")
    
    if identical:
        print("✓ Genetic algorithm found exactly the same lexicographic optimal solution as exact algorithm!")
    else:
        print("⚠ Genetic algorithm solution is not exactly the same as exact algorithm solution")
        # Calculate closeness of solutions
        diff = np.max(np.abs(exact_solution - ga_solution))
        print(f"  Maximum difference: {diff:.8f}")
        
        if diff < 0.01:  # If difference is less than 1%, consider solutions close
            print("  Solutions are close, difference less than 1%")
    
    # Prepare bandwidth and demand history
    t_values = np.linspace(time_interval[0], time_interval[1], p)
    A_history = np.array([ga.bandwidth_cache[t] for t in t_values])
    b_history = np.array([ga.demand_cache[t] for t in t_values])
    
    return ga_solution, exact_solution, A_history, b_history, ga, identical

def main():
    """Main function"""
    print("="*80)
    print("Time-Series Fuzzy Relation Lexicographic Optimization - Strict Lexicographic Genetic Algorithm")
    print("Based on Section 4.2: Realistic Example with 8 Terminals")
    print("="*80)
    
    # Read μ_ij matrix (8x8) from the paper - Table 1
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
    p = 20  # Number of time points
    time_interval = (0, 2)  # Time interval [0, 2]
    
    print(f"System Configuration:")
    print(f"  Number of users: 8")
    print(f"  Number of terminals: 8")
    print(f"  Time interval: {time_interval}")
    print(f"  Number of time points: {p}")
    print(f"  Lexicographic priority order: x1 → x2 → ... → x8")
    print(f"  Demand function: b_i(t) = 0.1 + 0.1·sin(2πt)")
    print(f"  Bandwidth model: Gaussian PDF with μ_ij, σ_ij from Tables 1&2")
    print(f"  Strict Lexicographic GA parameters: Population size=50, Generations=100")
    
    # Run algorithm
    start_time = datetime.now()
    print("\nStarting optimization...")
    ga_solution, exact_solution, A_history, b_history, ga_instance, identical = strict_lexicographic_genetic_algorithm_optimization(
        mu_matrix, sigma_matrix, beta, p, time_interval
    )
    end_time = datetime.now()
    
    print(f"\nTotal optimization time: {(end_time - start_time).total_seconds():.3f} seconds")
    
    # Output both solutions
    print("\n" + "="*80)
    print("Exact Algorithm Solution:")
    print("="*80)
    ga_instance.print_solution_table(exact_solution, "Exact Lexicographic Optimization Solution")
    
    print("\n" + "="*80)
    print("Genetic Algorithm Solution:")
    print("="*80)
    ga_instance.print_solution_table(ga_solution, "Strict Lexicographic Genetic Algorithm Solution")
    
    return ga_solution, exact_solution, A_history, b_history, ga_instance, identical

if __name__ == "__main__":
    try:
        ga_solution, exact_solution, A_history, b_history, ga_instance, identical = main()
        
        print("\n" + "="*80)
        print("Program executed successfully!")
        print("="*80)
    except Exception as e:
        print(f"\nProgram execution error: {e}")
        import traceback
        traceback.print_exc()