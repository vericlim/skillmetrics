import xarray as xr


def pca_xarray(data, n_components,dims_eof,dims_pc):
    """
    Calculates Principal Component Analysis 
    
    parameters
    ----------
    data : Data field with at least two dimensions. 
    n_components: Number of components calculated
    dims_eof: List of dimensions over which Empirical Orthogonal Functions should be calculated
    dims_pc : List of dimensions over which Principal Components are analysed 
    
    returns
    -------
    Empirical orthogonal functions
    Principal Components
    """
    
    pca = PCA(n_components = n_components)
    
    data_stacked = data.stack(pseudo_time = dims_pc).stack(location=dims_eof)
    
    N_location    = data_stacked.sizes["location"]
    N_pseudo_time = data_stacked.sizes["pseudo_time"]
    
    data_stacked_transpose_dropped = data_stacked.transpose("pseudo_time","location").dropna(dim="location")
    
    projected= pca.fit_transform(data_stacked_transpose_dropped.values)

    
    explained_variance = pca.explained_variance_ratio_
    EOF = xr.DataArray(pca.components_              , dims=("component","location")   , coords = (np.arange(n_components),data_stacked_transpose_dropped.location)).unstack(dim="location")
    PCs = xr.DataArray(projected                    , dims=("pseudo_time","component"), coords = (data_stacked_transpose_dropped.pseudo_time,np.arange(n_components))).unstack(dim="pseudo_time")
    EXP = xr.DataArray(pca.explained_variance_ratio_, dims=("component",)             , coords = (np.arange(n_components),))
    
    return xr.merge([EOF.rename("EOF"),PCs.rename("PCs"),EXP.rename("EXP")])