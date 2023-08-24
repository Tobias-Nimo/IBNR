import numpy as np
import pandas as pd

def import_triangle(path, sheet, as_=("triangle", "table")):
    """imports triangle from excel file as df (triangle or table)"""
    
    loss_triangle = pd.read_excel(path, sheet_name=sheet, index_col=0)
    loss_table = loss_triangle.stack().reset_index().rename(columns={'level_0': 'acc_year',
                                                                     'level_1': 'dev_year',
                                                                     0: 'loss'})
    
    if as_ == "triangle":
        return loss_triangle
    elif as_ == "table": 
        return loss_table
    else:
        return None

def calculate_stuff(X): # X is design matrix (without col of 1s for intercept!!!)
    """takes sm design matrix as input,
    computes constants of interest related to the shape of the triangle"""
    n = X.shape[0] 
    p = X.shape[1]+1 # +1 because of the constant

    lim, count = 1, 0
    while count<n:
        lim+=1
        count = sum([i for i in range(1,lim)])
    t = lim-1 
    
    # n: number of known elements in the upper triangle
    # p: number of params to estimate
    # t: number of acc./dev. years
    return (n, p, t)

def gen_formula(X):
    """given number of parameters to estimate, generates GLM formula"""
    n, p, t = calculate_stuff(X)
    
    formula="loss ~"
    _ = int((p-1)/2)

    for s in range(1,_+1):
        formula += f' acc_year_{s+1} +'
    for s in range(1,_+1):
        formula += f' dev_year_{s+1} +'
    formula = formula[:-2]
    
    return formula

def model_eq(i, j, model):
    """Given coordinate-pair (i,j), evaluates model equation"""
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names).drop(columns="Intercept")
    n, p, t = calculate_stuff(X)
    link = model.family.link.__class__.__name__.lower()
    parameters = model.params
    
    # intercept
    c = parameters[0] 
    # acc. years factors
    if i == 1:
        alpha = 0
    else:
        alpha = parameters[i-1] 
    # dev. years factors
    if j == 1:
        beta = 0
    else:
        beta = parameters[j+t-2] 
    
    # link function
    if link == 'log' or link == 'Log':
        return np.exp(c + alpha + beta)
    elif link == 'inverse_squared' or link == 'inversesquared':
        return (c + alpha + beta)**(-1/2)
    
def gen_adj_table(model):
    """given a model, returns populated loss table with estimated/adjusted incremental values"""
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names).drop(columns="Intercept")
    n, p, t = calculate_stuff(X)
    
    adj_data = []
    for i in range(1,t+1):
        for j in range(1,t+1):
            adj_data.append((i, j, model_eq(i, j, model)))
    adj_data = pd.DataFrame(adj_data, columns=['AccYear', 'DevYear', 'Loss*'])
    return adj_data

def to_triangular_form(tabular_data):
    """takes a df loss data and transforms it into a triangle"""
    triangle = tabular_data.pivot_table(index="AccYear", 
                                        columns="DevYear", 
                                        values=tabular_data.columns[2], # "Loss*"
                                        fill_value=np.nan).reset_index(drop=True).rename_axis(None, axis=1)
    t = triangle.shape[0]
    triangle.index.name = None
    triangle.index = pd.RangeIndex(start=1, stop=t+1, step=1)
    return triangle
    
def to_cum_triangle(inc_triangle):
    """takes incremental triangle as input, returns cumulative triangle"""
    return np.cumsum(inc_triangle, axis=1)

def calculate_loss(complete_cum_triangle):
    """Given cumulative loss matrix (upper+lower triangle),
    computes ultimates loss and IBNR reserve."""
    t = complete_cum_triangle.shape[1]
    
    ultimate = np.array(complete_cum_triangle.iloc[:,t-1])
    anti_diag = np.flip(np.array(complete_cum_triangle)[::-1,:].diagonal())
    ibnr = ultimate - anti_diag 
    
    ultimate = pd.Series(ultimate, index=range(1,t+1))
    ibnr = pd.Series(ibnr, index=range(1,t+1))
    
    return ultimate, ibnr



