import numpy as np
THRESHOLD = 1e-8  # Threshold for floating point comparison

class Raytracer:
    def __init__(self, dimensions, start_coords, end_coords):
        # Initialize the raytracer
        self.initialize(dimensions, start_coords, end_coords)
    
    def initialize(self, dimensions, start_coords, end_coords):
        """Initialize all raytracing variables."""
        # Input parameters
        self.dimensions = dimensions
        
        # Convert coordinates to float arrays (handles both integer and float inputs)
        self.start_coords = np.array(start_coords, dtype=float)
        self.end_coords = np.array(end_coords, dtype=float)
        
        # Ray parameters
        self.x0 = self.start_coords.copy()  # start coordinates
        self.xf = self.end_coords.copy()    # goal coordinates  
        self.delta_x = self.xf - self.x0    # Δx (direction vector)
        self.ray_length = np.sqrt(np.sum(self.delta_x**2))  # ||Δx|| (total ray length)
        
        # Raytracing state variables
        self.delta_x_sign = np.zeros(self.dimensions, dtype=int)    # δx (direction signs: -1, 0, 1)
        self.D = np.zeros(self.dimensions, dtype=float)             # D (parametric distances to next grid lines)
        self.D0 = np.zeros(self.dimensions, dtype=float)            # D0 (initial D values backup)
        self.y = np.zeros(self.dimensions, dtype=int)               # y (current corner coordinates)
        self.k = np.zeros(self.dimensions, dtype=int)               # k (number of grid crossings per dimension)
        self.t = 0.0                                                # t (current parametric position along ray)
        self.F = []                                                 # F (front cell relative coordinates matrix)
        
        # Validate input coordinates
        if not self._validate_coordinates():
            return
        
        # Calculate initial D and y values
        for i in range(self.dimensions):
            if self.delta_x[i] > THRESHOLD:
                # Moving in positive direction
                self.delta_x_sign[i] = 1
                self.y[i] = int(np.floor(self.x0[i]))
                self.D[i] = (self.y[i] - self.x0[i] + self.delta_x_sign[i]) / self.delta_x[i]
            elif self.delta_x[i] < -THRESHOLD:
                # Moving in negative direction
                self.delta_x_sign[i] = -1
                self.y[i] = int(np.ceil(self.x0[i]))
                self.D[i] = (self.y[i] - self.x0[i] + self.delta_x_sign[i]) / self.delta_x[i]
            else:
                # Not moving in this dimension (Δxi = 0)
                self.delta_x_sign[i] = 0
                self.y[i] = int(np.floor(self.x0[i]))
                self.D[i] = float('inf')
            
            # Handle edge case where D is very close to 0
            if abs(self.D[i]) < THRESHOLD and abs(self.delta_x[i]) > THRESHOLD:
                self.D[i] = 1.0 / abs(self.delta_x[i])
        
        # Store initial D values
        self.D0 = self.D.copy()

    def coords(self):
        if self.ray_length == 0:
            return self.x0.copy()
        return self.x0 + (self.length() / self.ray_length) * self.delta_x

    def front_cells(self):
        if self.reached():
            return []
        
        # Initialize F matrix to store front cell relative coordinates
        self.F = []
        
        # Start recursive generation with initial sign vector
        sgn_diff_coords = self.delta_x_sign.copy()
        self._getF(0, sgn_diff_coords.copy())  # Start from dimension 0 (0-indexed)
        
        # Generate actual front cell coordinates: x_c(t) = y(t) + f_j
        front_cells = []
        for f in self.F:
            cell_coord = self.y + f
            front_cells.append(cell_coord)
        
        return front_cells
    
    def _getF(self, d, sgn_diff_coords):
        if d >= self.dimensions:  # d >= MAX_DIM
            self.F.append(sgn_diff_coords.copy())
            return
        
        is_delta_x_zero = abs(self.delta_x[d]) < THRESHOLD
        is_x0_integer = abs(self.start_coords[d] - round(self.start_coords[d])) < THRESHOLD
        
        if is_delta_x_zero and is_x0_integer:
            sgn_diff_coords[d] = -1
            self._getF(d + 1, sgn_diff_coords.copy())

            sgn_diff_coords[d] = 0
            self._getF(d + 1, sgn_diff_coords.copy())
        else:
            if self.delta_x_sign[d] < 0:
                sgn_diff_coords[d] = -1
            else:
                sgn_diff_coords[d] = 0
            self._getF(d + 1, sgn_diff_coords.copy())

    def length(self):
        return self.t * self.ray_length

    def reached(self):
        return self.t >= 1.0

    def next(self):
        if self.reached():
            return False
        
        # Find the dimension with minimum D value
        i = np.argmin(self.D)
        
        # Update parameters
        self.t = self.D[i]
        
        # Update current position to the new ray position
        self.x0 = self.start_coords + self.t * self.delta_x

        # Find other dimensions that are close to the minimum D value
        for j, d_j in enumerate(self.D):
            if abs(d_j - self.t) < THRESHOLD:
                if self.delta_x_sign[j] == 0:
                    continue

                self.y[j] = self.y[j] + self.delta_x_sign[j]
                self.k[j] = self.k[j] + 1
                self.D[j] = self.D0[j] + (self.k[j] / abs(self.delta_x[j])) 
        
        return True
    
    def trace(self):
        intersected_cells = set()  # Use set to avoid duplicates
        
        # Traverse the ray and collect all front cells at each step
        while not self.reached():
            # Get front cells at current position
            current_front_cells = self.front_cells()
            for cell in current_front_cells:
                intersected_cells.add(tuple(cell))
            
            # Move to next grid crossing
            if not self.next():
                break
        
        # Collect front cells at the final position (goal node)
        final_front_cells = self.front_cells()
        for cell in final_front_cells:
            intersected_cells.add(tuple(cell))
        
        return list(intersected_cells)
    
    def _validate_coordinates(self):
        """Validate input coordinates."""
        # Check dimensions
        if len(self.start_coords) != self.dimensions or len(self.end_coords) != self.dimensions:
            return False
        return True
