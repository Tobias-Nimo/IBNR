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

def to_cum_triangle(inc_triangle):
    """takes incremental triangle as input, returns cumulative triangle"""
    return np.cumsum(inc_triangle, axis=1)

def compute_dev_factors(cum_triangle):
    """given a cumulative loss triangle, computes development factors"""

    t = cum_triangle.shape[1] # number of acc/dev years
    dev_factors = []
    i = 1
   
    for j in range(t):
        try:
            f = cum_triangle.iloc[:,j+1].sum()/cum_triangle.iloc[:-i,j].sum()
            dev_factors.append(f)
        except IndexError:
            pass
        i+=1
    dev_factors.append(1) # tail factor
    
    return np.array(dev_factors)
    
def chain_ladder(i, j, cum_triangle):
    
    # development factors
    dev_factors = compute_dev_factors(cum_triangle)
    
    # main anti-diagonal
    anti_diagonal = np.array(cum_triangle)[::-1, :].diagonal()
    
    # calculate adjusted/estimated cumulative value
    t = cum_triangle.shape[1]
    cum_loss = anti_diagonal[-i]
    if i+j<=(t+1):
        F = dev_factors[j-1:-i].prod()
        return  cum_loss/F
    else:
        h = i+j-(t+1)
        F = dev_factors[-i:-i+h].prod()
        return cum_loss*F
    
def gen_adj_table(cum_triangle):
    """given a cumulative triangle,
    populates a loss data df with adjusted/estimated values with chain ladder"""
    adj_data = []
    t = cum_triangle.shape[1]
    for i in range(1,t+1):
        for j in range(1,t+1):
            adj_data.append((i, j, chain_ladder(i, j, cum_triangle)))

    adj_data = pd.DataFrame(adj_data, columns=['acc_year', 'dev_year', 'loss'])
    return adj_data

def to_triangular_form(tabular_data):
    """takes a df loss data and transforms it into a inc/cum triangle"""
    triangle = tabular_data.pivot_table(index="acc_year", 
                                        columns="dev_year", 
                                        values=tabular_data.columns[2], # loss
                                        fill_value=np.nan).reset_index(drop=True).rename_axis(None, axis=1)
    t = triangle.shape[0]
    triangle.index.name = None
    triangle.index = pd.RangeIndex(start=1, stop=t+1, step=1)
    return triangle
    
def to_inc_triangle(cum_triangle):
    """takes cumulative triangle, returns incremental triangle"""
    inc_triangle = cum_triangle.diff(axis=1)
    inc_triangle.iloc[:,0] = cum_triangle.iloc[:,0]
    
    return inc_triangle
    
def calculate_loss(complete_cum_triangle, tail=1):
    """given cumulative loss matrix (upper+lower triangle),
    computes ultimates loss and IBNR reserve."""
    t = complete_cum_triangle.shape[1]
    
    acum_dev = np.array(complete_cum_triangle.iloc[:,t-1])
    ultimate = acum_dev*tail
    anti_diag = np.flip(np.array(complete_cum_triangle)[::-1,:].diagonal())
    ibnr = ultimate - anti_diag 
    
    ultimate = pd.Series(ultimate, index=range(1,t+1))
    ibnr = pd.Series(ibnr, index=range(1,t+1))
    
    return ultimate, ibnr