import numpy as np
from MLmodel import MLModel
from SageFunctions_ import SageFunctions, list2vertices
import pygad
from GAPopulationFuncs import generateInitialPopulation, mutation_func, crossover_func, population2populationVertices, reduce_population, vertices2MagmaStr
from sage.all import *
import ast
import sklearn
from sklearn.metrics import *
import pandas as pd
import os
from ToricFunctions import analyse_datasets, explorer
SAGE_EXTCODE = SAGE_ENV['SAGE_EXTCODE']
magma.attach('%s/magma/sage/basic.m'%SAGE_EXTCODE)

# Transformer ML Model parameters
PATH = "transformer_3_attention_mean.pt"
args = None

# Initialize class which provides access to model
model = MLModel(PATH, model_type = "Transformer_v3")

# best distances over F7
bestdistances = (30, 27, 24, 24, 23, 20, 20, 18, 18, 17, 15, 15, 14, 12, 12, 12, 12, 10, 10, 9, 8, 8, 6, 6, 6, 6, 5, 4, 4, 3, 3)

prime = magma.eval(Integer(7))
primitive = magma.eval('PrimitiveElement(FiniteField(' + prime +'))')
allvertices = magma.eval('[<x,y> : x,y in [0..'+prime+'-2]]')
print(allvertices)
generalised_toric_matrix = magma.eval('generalisedToricMatrix := function(vertices); M := KMatrixSpace(FiniteField('+prime+'), #vertices, ('+prime+'-1)^2); rows := ['+primitive+'^(('+allvertices+'[j][1])*vertices[i][1] + ('+allvertices+'[j][2])*vertices[i][2]): j in [1..('+prime+'-1)^2], i in [1..#vertices]]; toricmatrix := M ! rows; return toricmatrix; end function;')

total_runs = 29

if total_runs == 0:
    for number in range(3,34): # change range to explore certain dimensions
        explorer(number, total_runs, set())
else:
    all_found, approximate_distances, actual_distances, knowndistances = analyse_datasets(total_runs)
    for i in range(0,len(all_found)):
        print(all_found[i])
    while not np.array_equal(knowndistances, bestdistances):
        for number in range(3,34): # change range to explore certain dimensions
            if knowndistances[number-3] != bestdistances[number-3]:
                explorer(number, total_runs, all_found)
        all_found, approximate_distances, actual_distances, found_distances = analyse_datasets(total_runs)    
        knowndistances = found_distances
        total_runs += 1