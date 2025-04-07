from random import randint, seed
from sage.all import preparser, set_random_seed, Integer, MatrixSpace, GF
from sage.all import codes
from sage.matrix.constructor import matrix

import numpy as np

global_np_dtype = np.float32

preparser(False)

def set_seed(seed_val):
    set_random_seed(Integer(seed_val))
    seed(seed_val)

def list2vertices(input_list, ordered=True):
    if len(input_list) % 2 != 0:
        print("list2vertices: incorrect list length ", len(input_list))
        return None

    output_list = [ (input_list[2*i], input_list[2*i+1]) for i in range(len(input_list)//2) ]
    if ordered:
        output_list.sort()
    return output_list

def vertices2list(vertices):
    return [coord for vertex in vertices for coord in vertex] # flatten list of vertices

class SageFunctions():
    def __init__(self, prime):
        self.prime = prime

        self.F = GF(Integer(prime))  # generate finite field of order prime
        self.primitive = self.F.multiplicative_generator()  # Primitive element of the finite field
        self.allvertices = [(x, y) for x in range(prime - 2 + 1) for y in range(prime - 2 + 1)]  # Ordered set of all vertices, include p-2

    def generate_random_vertex(self):
        i = randint(0, self.prime - 2 + 1) # to include upper bound
        j = randint(0, self.prime - 2 + 1) # to include upper bound
        return (i, j)

    # Function to generate a random set of n_vert vertices
    def generate_random_vertices(self, n_vert):
        vertices = []
        while len(vertices) < n_vert:
            new_vertex = self.generate_random_vertex()
            if new_vertex not in vertices:
                vertices.append(new_vertex)
        vertices.sort()
        return vertices

    # Function to generate n_sets sets of vertices each with n_vert vertices
    def generate_sets_of_vertices(self, n_sets, n_vert):
        vertices_sets = []
        while len(vertices_sets) < n_sets:
            new_vertices = self.generate_random_vertices(n_vert)
            if new_vertices not in vertices_sets:
                vertices_sets.append(new_vertices)
        return list(vertices_sets)

    # Left for backward compatibility
    def list2vertices(self, input_list, ordered=False):
        return list2vertices(input_list, ordered)

    def toricMatrixRowsFromVertices_(self, vertices):
        primitive = self.F.multiplicative_generator()
        rows = []
        for v in vertices:
            row = [
                primitive ** Integer(self.allvertices[j][0] * v[0] + self.allvertices[j][1] * v[1])
                for j in range((self.prime - 1) ** 2)
            ]
            rows.append(row)

        return rows

    # Function to generate the toric matrix
    def vertices2toricMatrix_(self, vertices):
        k = Integer(len(vertices))
        n = Integer((self.prime - 1)**2)
        MS = MatrixSpace(GF(Integer(self.prime)), k, n)

        matrix_rows = self.toricMatrixRowsFromVertices_(vertices)
        toricmatrix = MS(matrix_rows)  # generate a finite field matrix in Sage  

        return toricmatrix
    
    def vertices2Generator(self, vertices):
        toric_matrix = self.vertices2toricMatrix_(vertices)
        basis = toric_matrix.row_space().basis()
        generator = matrix(self.F, basis)
        return np.array(generator, dtype=global_np_dtype)

    def vertices2code_(self, vertices):
        toric_matrix = self.vertices2toricMatrix_(vertices)
        toric_code = codes.LinearCode(toric_matrix)
        return toric_code

    # Function to calculate the minimum Hamming distance
    def vertices2mindist(self, vertices):
        code = self.vertices2code_(vertices)
        return code.minimum_distance()