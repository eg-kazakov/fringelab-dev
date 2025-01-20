import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import ConvexHull

from .base_class_correction_algorithm import CorrectionAlgorithm
from .. import FringeSet


class MultiFociCorrectionAlgorithm(CorrectionAlgorithm):
    LEMNISCATE = "l"
    ELLIPSE = "e"

    def __init__(self, mode=ELLIPSE, max_iterations=10, tolerance=1e-6, debug=False):
        self.mode = mode
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.debug = debug

    def recover_phase(self, I1, I2, I3):
        # Placeholder implementation
        fringe_set = FringeSet.create_set_from_fringes(I1, I2, I3)
        trajectory = fringe_set.get_trajectory()
        return self.recover_phase(trajectory)

    def recover_phase(self, points):

        # Step 1: Initial focus and radius
        z1, R = self._step1_init(points)
        print("init")
        self._plot_state(points, [z1], R)
        # Step 2: Optimize initial focus and radius
        z1, R, error = self._step2_optimize(z1, R, points)

        # Initialize foci
        foci = [z1]
        print("optimize")
        self._plot_state(points, foci, R)
        cur_error = None
        for iteration in range(self.max_iterations):
            if len(foci) >= 4:
                # Attempt to remove a focus to improve approximation
                foci_remove, R_remove, error_remove = self._attempt_remove_focus(foci, R, points)
                print(f"Attempted to remove a focus, error after removal: {error_remove}")
                self._plot_state(points, foci_remove, R_remove)
                if cur_error > error_remove:
                    foci = foci_remove
                    R = R_remove

            # Step 3: Double the foci
            foci, R = self._step3_multiply_foci(foci, R)

            print(f"multiply {iteration}")
            self._plot_state(points, foci, R)

            # Step 4: Foci motion step
            foci, R = self._step4_move_foci(foci, R, points)

            print(f"move {iteration}")
            self._plot_state(points, foci, R)

            # Step 5: Optimize parameters with convex hull constraint
            foci, R, total_error = self._step5_optimize(foci, R, points)

            print(f"optimize: {iteration}")
            print(f"total_error: {total_error}")
            self._plot_state(points, foci, R)

            # Step 6: Check for convergence
            if total_error < self.tolerance:
                break
            cur_error = total_error

        return foci, R

        # Function to compute convex hull constraints
    def _get_convex_hull_constraints(self, points, num_foci):
        hull = ConvexHull(points)
        # The convex hull in half-space representation: A x <= b
        A = hull.equations[:, :-1]  # Shape: (num_constraints, 2)
        b = -hull.equations[:, -1]  # Shape: (num_constraints,)

        num_constraints = A.shape[0]

        # Initialize total constraint matrices
        A_total = np.zeros((num_foci * num_constraints, num_foci * 2 + 1))
        b_total = np.tile(b, num_foci)

        # Build block-diagonal constraint matrix for all foci
        for i in range(num_foci):
            row_start = i * num_constraints
            row_end = (i + 1) * num_constraints
            col_start = i * 2
            col_end = (i + 1) * 2
            A_total[row_start:row_end, col_start:col_end] = A
            # The last column (for R) remains zeros

        return A_total, b_total

    # Modify step2_optimize to add convex hull constraints
    def _step2_optimize(self, z1, R, points):
        initial_params = np.hstack((z1, R))
        num_foci = 1
        A_total, b_total = self._get_convex_hull_constraints(points, num_foci)

        linear_constraint = LinearConstraint(
            A_total,
            -np.inf,
            b_total
        )

        result = minimize(
            self._error_function_single_focus,
            initial_params,
            args=(points,),
            method='SLSQP',
            constraints=[linear_constraint]
        )
        error = result.fun
        z1 = result.x[:2]
        R = result.x[2]
        print(result)
        return z1, R, error

    def _step5_optimize(self, foci, R, points):
        N = len(foci)
        initial_params = np.hstack((np.array(foci).flatten(), R))
        A_total, b_total = self._get_convex_hull_constraints(points, N)

        linear_constraint = LinearConstraint(
            A_total,
            -np.inf,
            b_total
        )

        result = minimize(
            self._error_function_multi_foci,
            initial_params,
            args=(points, N),
            method='SLSQP',
            constraints=[linear_constraint]
        )
        optimized_params = result.x
        foci = optimized_params[:-1].reshape(N, 2)
        R = optimized_params[-1]
        total_error = result.fun
        return foci.tolist(), R, total_error

    def _step1_init(self, points):
        z1 = np.mean(points, axis=0)
        distances = np.linalg.norm(points - z1, axis=1)
        R = self._get_radius(distances)
        return z1, R

    def _error_function_single_focus(self, params, points):
        z1_x, z1_y, R = params
        z1 = np.array([z1_x, z1_y])
        distances = np.linalg.norm(points - z1, axis=1)
        L = distances - R
        F = np.sum(L**2)
        return F

    def _error_function_multi_foci(self, params, points, N):
        foci = params[:-1].reshape(N, 2)
        R = params[-1]
        distances = np.linalg.norm(points[:, np.newaxis, :] - foci, axis=2)

        product_distances = np.sum(distances, axis=1) if self._ellipse_mode() else np.prod(distances, axis=1)
        L = product_distances - R
        F = np.sum(L**2)
        return F

    def _step3_multiply_foci(self, foci, R):
        foci.extend(foci.copy())
        R=R*2 if self._ellipse_mode() else R**2
        return foci, R

    def _step4_move_foci(self, foci, R, points):
        delta = 0.1  # Small constant for movement
        N = len(foci)
        foci_array = np.array(foci)

        for idx in range(N):
            min_error = None
            best_move = None
            directions = [np.array([delta, 0]), np.array([-delta, 0]), np.array([0, delta]), np.array([0, -delta])]
            for direction in directions:
                temp_foci = foci_array.copy()
                temp_foci[idx] += direction
                params = np.hstack((temp_foci.flatten(), R))
                error =self._error_function_multi_foci(params, points, N)
                if min_error is None or error < min_error:
                    min_error = error
                    best_move = direction
            foci_array[idx] += best_move
        distances = np.linalg.norm(points[:, np.newaxis, :] - foci_array, axis=2)
        product_distances = np.prod(distances, axis=1)
        R = np.exp(np.mean(np.log(product_distances)))
        return foci_array, R


    def _estimate_radius(self, foci, R, points):
        # Implement logic to estimate R based on the new set of foci and points
        # This could be based on previous R or re-initialized
        # For simplicity, you can use the previous R
        return R  # Placeholder; adjust as necessary


    def _attempt_remove_focus(self, foci, R, points):
        best_error = float('inf')
        best_foci = None
        best_R = None

        for i in range(len(foci)):
            # Create a new set of foci without the i-th focus
            foci_candidate = foci[:i] + foci[i+1:]

            # Estimate new R based on the reduced number of foci
            R_candidate = self._estimate_radius(foci_candidate, R, points)

            # Optimize parameters for the candidate foci
            foci_candidate_opt, R_candidate_opt, error_candidate = self._step5_optimize(
                foci_candidate, R_candidate, points)

            if error_candidate < best_error:
                best_error = error_candidate
                best_foci = foci_candidate_opt
                best_R = R_candidate_opt

        return best_foci, best_R, best_error

    @staticmethod
    def _lemniscate_function(x, y, foci, R):
        result = 1.0
        for focus in foci:
            result *= np.sqrt((x - focus[0])**2 + (y - focus[1])**2)
        return result - R

    @staticmethod
    def _ellipse_function(x, y, foci, R):
        result = 0.0
        for focus in foci:
            result += np.sqrt((x - focus[0])**2 + (y - focus[1])**2)
        return result - R

    def _plot_state(self, points, foci, R):
        if not self.debug:
            return
        foci = np.array(foci)
        print(R, foci)
        # Create the plot
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, color='blue', marker='o', label='Data Points')
        plt.scatter(foci[:, 0], foci[:, 1], color='red', marker='x', s=100, label='Foci')


        # Create a grid for plotting the lemniscate curve
        grid_size = 500
        x_min, x_max = foci[:, 0].min() - 1, foci[:, 0].max() + 1
        y_min, y_max = foci[:, 1].min() - 1, foci[:, 1].max() + 1

        print(x_min, x_max)
        print(y_min, y_max)
        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min, x_max, grid_size),
            np.linspace(y_min, y_max, grid_size)
        )

        # Compute the ellipsoid values on the grid
        if self._ellipse_mode():
            z_values = self._ellipse_function(x_grid, y_grid, foci, R)
        else:
            z_values = self._lemniscate_function(x_grid, y_grid, foci, R)

        # Plot the lemniscate curve
        plt.contour(x_grid, y_grid, z_values, levels=[0], colors='red', linewidths=2, label='Lemniscate Curve')


        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Visualization of Initial Set of Points, Initial Center, and Initial Circle')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.axis('equal')  # Equal scaling for both axes

        # Display the plot
        plt.show()

    @staticmethod
    def _get_lemniscate_radius(distances):
        return np.exp(np.mean(np.log(distances)))

    @staticmethod
    def _get_ellipse_radius(distances):
        return np.mean(distances)

    def _get_radius(self, distances):
        if self._ellipse_mode():
            return self._get_ellipse_radius(distances)
        else:
            return self._get_lemniscate_radius(distances)

    def _ellipse_mode(self):
        return self.mode == self.ELLIPSE