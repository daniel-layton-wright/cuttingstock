import numpy as np
from typing import Tuple
from scipy.optimize import linprog, LinearConstraint, Bounds, milp, OptimizeResult


class CuttingStockInput:
    def __init__(self, L, orders):
        self.L = L
        
        # Sort orders descending by length
        orders.sort(key=lambda x: x[0], reverse=True)
        
        self.orders = orders
        self.l = np.array([l for l, d in orders])
        self.d = np.array([d for l, d in orders])
        

class CuttingStockOutput:
    def __init__(self, input: CuttingStockInput, A):
        self.input = input
        self.A = A
        self.theta = None
        self.pi = None
        self.solve_dual()
        
    def solve_dual(self):
        """
        we'll solve for the dual problem, which is:
        max sum pi_i * d_i = min - sum pi_i * d_i
        s.t. sum_i pi_i * A_ip <= 1 for all p
        pi_i >= 0 for all i
        """
        res = linprog(-self.input.d, A_ub=self.A.T, b_ub=np.ones(self.A.shape[1]), method='highs')

        if res.success:
            self.pi = res.x
        else:
            raise ValueError("Failed to solve the linear program")
        
    def solve_primal(self, time_limit: int = 60):
        """
        Solve the integer primal program:
        
        min sum theta_i
        s.t. A * theta >= d
            theta >= 0
        """
        if self.theta is not None:
            return
        
        c = np.ones(self.A.shape[1])
        
        constraints = LinearConstraint(self.A, lb=self.input.d, ub=np.inf)
        
        res = milp(c=c, constraints=constraints, integrality=np.ones_like(c), options={'disp': True, 'time_limit': time_limit})
        
        if res.status == 1:
            import warnings
            warnings.warn(f"Time limit reached. Gap at termination: {res.mip_gap}")
        
        if res.status not in [0, 1]:
            raise ValueError("Failed to solve the primal linear program")
        
        self.theta = res.x
        
    def remove_unused_columns(self):
        """
        Remove columns from A that are not used in the solution
        """
        self.A = self.A[:, self.theta > 0]
        self.theta = self.theta[self.theta > 0]
        
    def __str__(self):
        return (f"l: {self.input.l}\n"
                f"d: {self.input.d}\n"
                f"A ({self.A.shape[0]} x {self.A.shape[1]}):\n{np.round(self.A)}\n"
                f"pi: {np.round(self.pi, 2)}")


def read_input(input_file: str) -> CuttingStockInput:
    with open(input_file, 'r') as file:
        lines = file.readlines()
        
    input = CuttingStockInput(int(lines[0]), [tuple(map(int, line.split())) for line in lines[1:]])

    return input


def generate_initial_solution(input: CuttingStockInput) -> CuttingStockOutput:
    """
    This generates a very naive initial solution by generating patterns sequentially. 
    """
    A_columns = []
    
    new_column = lambda: np.zeros(len(input.orders))
    
    for i, (l, d) in enumerate(input.orders):
        # max number of times l will fit in L
        k = min(input.L // l, d)
        cur_column = new_column()
        cur_column[i] = k
        A_columns.append(cur_column)
            
    A = np.column_stack(A_columns)
    
    return CuttingStockOutput(input, A)


def get_minimum_reduced_cost(solution: CuttingStockOutput) -> Tuple[float, np.ndarray]:
    """
    We need to solve the following integer program:
    
    We need to solve the following integer program:
    
    Minimize: 1 - sum(pi[i] * x[i] for i in Orders)
    Equivalent to maximize: sum(pi[i] * x[i] for i in Orders)
    Subject to:
        sum(l[i] * x[i] for i in Orders) <= L
        x[i] <= d[i] for all i in Orders
        x[i] >= 0 and integer for all i in Orders
    
    Where:
    x[i] = number of times order i is cut in this pattern
    pi[i] = dual variable for order i
    l[i] = length of order i
    d[i] = demand for order i
    L = total length of stock
    Orders = set of all orders
    """    
    # Define the objective function coefficients (pi values)
    c = -solution.pi
    constraints = LinearConstraint(np.array([solution.input.l]), np.array([0]), np.array([solution.input.L]))
    
    # Define the bounds for each variable
    bounds = Bounds(0, solution.input.d)
        
    # Solve the integer program
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones_like(c))

    # Calculate the reduced cost
    reduced_cost = 1 + np.dot(c, res.x)
    
    # Return the reduced cost and the new column
    return reduced_cost, res.x


def add_column(solution: CuttingStockOutput, new_column: np.ndarray) -> CuttingStockOutput:
    return CuttingStockOutput(solution.input, np.column_stack([solution.A, new_column]))


def print_results(i, cur_solution, minimum_reduced_cost, new_column):
    print('#' * 100)
    print(f'Iteration {i}:')
    print(f'Current solution:\n{cur_solution}')
    print(f'Minimum reduced cost: {minimum_reduced_cost}')
    if minimum_reduced_cost < 0:
        print(f'New column:\n{new_column}')
    else:
        print('OPTIMAL SOLUTION FOUND')
        print('#' * 100)


def do_column_generation(input: CuttingStockInput) -> CuttingStockOutput:
    i = 0
    cur_solution = generate_initial_solution(input)   
    minimum_reduced_cost, new_column = get_minimum_reduced_cost(cur_solution)
    
    print_results(i, cur_solution, minimum_reduced_cost, new_column)
    
    while minimum_reduced_cost < 0:
        i += 1
        cur_solution = add_column(cur_solution, new_column)
        minimum_reduced_cost, new_column = get_minimum_reduced_cost(cur_solution)
        print_results(i, cur_solution, minimum_reduced_cost, new_column)
    
    return cur_solution


def solve(input: CuttingStockInput, time_limit) -> CuttingStockOutput:
    solution = do_column_generation(input)
    solution.solve_primal(time_limit=time_limit)
    solution.remove_unused_columns()
    return solution


def write_output(solution: CuttingStockOutput, output_file: str):    
    with open(output_file, 'w') as file:
        # Write the objective function value
        file.write(f'{sum(solution.theta)}\n')
        
        A = np.round(solution.A)
        
        # Write the value of theta_p for each pattern p
        max_width = max(
            max(len(str(int(x))) for x in solution.theta),
            max(len(str(int(x))) for row in A for x in row)
        )
        theta_str = ' '.join(f'{int(x):>{max_width}}' for x in solution.theta)
        file.write(theta_str + '\n')
        
        # Write a blank line
        file.write('\n')
        
        # Write the matrix A
        for row in A:
            row_str = ' '.join(f'{int(x):>{max_width}}' for x in row)
            file.write(row_str + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument("output_file", type=str, help="Output file")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Time limit for solving the integer primal problem (only done once column generation is complete)")
    parser.add_argument("--debug", action="store_true", help="Open PDB at the end of the program")
    args = parser.parse_args()
    
    input = read_input(args.input_file)
    solution = solve(input, args.time_limit)
    write_output(solution, args.output_file)
    
    if args.debug:
        import pdb; pdb.set_trace()

    
if __name__ == "__main__":
    main()
