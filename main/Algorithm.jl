include("Graph.jl")
include("Tools.jl")

using SparseArrays
using Laplacians
using LinearAlgebra


#aprrox expressed opinion on original graph
function Approx(G, s; eps=1e-6)
    T = time()
    IpL = getSparseIpL(G)
    sL = getSparseL(G)
    f = approxchol_sddm(IpL, tol=0.1 * eps)
    n = G.n
    v = ones(n)
    s_mean = s - ((s' * v) / n) * v

    z_mean = f(s_mean)  # observation 1 in Mosco's paper
    p = z_mean' * z_mean
    pd = s_mean' * z_mean
    T = time() - T
    return T, p, pd
end

# compute pd index after adding C weights without approximation
function Exact_PD_index_CW_weighted(G, X, Y, s, W, C; eps=1e-6)
    n, k = size(X)
    w = (C * W) / (2 * n)
    v = ones(n)
    s_mean = s - ((s' * v) / n) * v

    T = time()

    A = getSparseIpLpDs_CW_weighted(G, X, Y, W, C)

    z_mean = inv(A - w * (X * Y + Y' * X')) * s_mean

    p = z_mean' * z_mean

    pd = s_mean' * z_mean

    T = time() - T
    return T, p, pd
end

function compute_lower_bound(G, s, W, C)
    T = time()
    n = G.n
    w_ = ((C * W) / (n * (n - 1) / 2))
    M = -w_ * ones(n, n)
    for i in 1:n
        M[i, i] = w_ * (n - 1)
    end

    w = (C * W) / (2 * n)
    v = ones(n)
    s_mean = s - ((s' * v) / n) * v
    IpLG = getIpL(G)
    z_mean = inv(IpLG + M) * s_mean
    p = z_mean' * z_mean
    pd = s_mean' * z_mean
    T = time() - T
    return T, p, pd
end

function Approx_M_CW_weigted(A, U, V, IK_inv; eps=1e-6)

    f = approxchol_sddm(A, tol=0.1 * eps)
    n, d = size(U)

    approx_M = zeros(n, d)

    for i = 1:d
        zi = collect(U[:, i])
        t = f(zi)
        for j = 1:n
            approx_M[j, i] = t[j]
        end
    end
    M = IK_inv + V * approx_M
    return M
end

# apply approximation algorithm to compute pd index with C weights 
function Approximation_PD_index_CW_weigted(G, X, Y, s, W, C; eps=1e-6)
    T = time()

    n, k = size(X)
    v = ones(n)
    s_mean = s - ((s' * v) / n) * v

    w = -(2 * n) / (C * W)
    U = [X Y']
    V = [Y; X']

 
    IK = Matrix{Float64}(I, 2k, 2k)
    IK_inv = w * IK 

    A = getSparseIpLpDs_CW_weighted(G, X, Y, W, C) 

    f = approxchol_sddm(A, tol=0.1 * eps)
    s1 = f(s_mean)
    s2 = V * s1
    M = Approx_M_CW_weigted(A, U, V, IK_inv)  
    s3 = inv(M) * s2
    s4 = U * s3
    s5 = f(s4)
    az = s1 - s5
    p = az' * az
    pd = s_mean' * az
    T = time() - T
    return T, az, p, pd 
end

