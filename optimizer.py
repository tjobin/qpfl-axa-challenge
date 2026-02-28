from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit.primitives import StatevectorSampler

def minimize_binary_loss(qp: QuadraticProgram):
    """
    Minimizes a loss function defined over a binary vector x using QAOA.
    
    Args:
        qp (QuadraticProgram): The objective function and constraints.
        
    Returns:
        OptimizationResult: Contains the optimal binary vector x and the minimum cost.
    """
    # 1. Define a classical optimizer to tune the continuous QAOA parameters
    optimizer = COBYLA(maxiter=100)
    
    # 2. Define the quantum execution primitive 
    sampler = StatevectorSampler()
    
    # 3. Instantiate the QAOA solver
    # reps defines the circuit depth (p-layers of Cost and Mixer Hamiltonians)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2)
    
    # 4. Wrap QAOA in the MinimumEigenOptimizer to handle the QUBO-to-Hamiltonian mapping
    qaoa_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
    
    # 5. Solve the binary optimization problem
    result = qaoa_optimizer.solve(qp)
    
    return result

# --- Example Usage ---
if __name__ == "__main__":
    # Define a simple binary optimization problem
    problem = QuadraticProgram()
    
    # Add binary variables x_0, x_1, x_2
    problem.binary_var(name="x_0")
    problem.binary_var(name="x_1")
    problem.binary_var(name="x_2")
    
    # Define the loss function to minimize: f(x) = -2x_0*x_1 + x_1*x_2 - x_0 + 2x_2
    problem.minimize(
        linear={"x_0": -1, "x_2": 2},
        quadratic={("x_0", "x_1"): -2, ("x_1", "x_2"): 1}
    )
    
    # Run the function
    optimal_solution = minimize_binary_loss(problem)
    
    print(f"Optimal binary vector x: {optimal_solution.x}")
    print(f"Minimum loss value: {optimal_solution.fval}")