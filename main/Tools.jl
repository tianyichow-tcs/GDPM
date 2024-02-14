include("Graph.jl")


using CSV
using DataFrames
using Random
using SparseArrays
using Distributions
using StatsBase




# generate synthetic  opinions in four different distributions 
function powerLaw(n; alp=2.5, xmin=1)
    Random.seed!(round(Int, time() * 10000))
    x = rand(n)
    for i = 1:n
        x[i] = xmin * ((1 - x[i])^(-1.0 / (alp - 1.0)))
    end
    xm = x[argmax(x)]
    x ./= xm
    return x
end

function Uniform(n)
    Random.seed!(round(Int, time() * 10000))
    x = rand(n)
    return x
end


function Exponential(n; lmd=1, xmin=1)
    Random.seed!(round(Int, time() * 10000))
    x = rand(n)
    for i = 1:n
        x[i] = xmin - (1.0 / lmd) * log(1 - x[i])
    end
    xm = x[argmax(x)]
    x ./= xm
    return x
end

function Polarized(n)
    n_half = floor(Int, n / 2)
    s_exp_re = -1 * (Exponential(n_half, lmd=2, xmin=0.1) - ones(n_half))
    s = cat(Exponential(n - n_half, lmd=2, xmin=0.1), s_exp_re, dims=1)
    return s
end


function getW(G)
    L = zeros(G.n, G.n)
    for (ID, u, v, w) in G.E
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    end
    for i = 1:G.n
        L[i, i] += 1.0
    end
    return inv(L)
end

function getIpL(G)
    L = zeros(G.n, G.n)
    for (ID, u, v, w) in G.E
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    end
    for i = 1:G.n
        L[i, i] += 1.0
    end
    return L
end

function getL(G)
    L = zeros(G.n, G.n)
    for (ID, u, v, w) in G.E
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    end
    return L
end


function getIpLpDs(G, X, Y)
    IpL = getIpL(G)
    S = X' * Y + Y' * X
    d, n = size(X)
    Ds = zeros(n, n)
    for i = 1:n
        t = 0
        for j in 1:n
            t += S[i, j]
        end
        Ds[i, i] = t
    end
    A = IpL + Ds
    return A
end

function getSparse_D_squre_root(G)
    n = G.n
    d = zeros(n)
    for (ID, u, v, w) in G.E
        d[u] += w
        d[v] += w
    end

    Is = zeros(Int32, G.m * 2 + G.n)
    Js = zeros(Int32, G.m * 2 + G.n)
    Vs = zeros(G.m * 2 + G.n)
    for (ID, u, v, w) in G.E
        Is[ID] = u
        Js[ID] = v
        Vs[ID] = 0
        Is[ID+G.m] = v
        Js[ID+G.m] = u
        Vs[ID+G.m] = 0
    end
    for i = 1:G.n
        Is[G.m+G.m+i] = i
        Js[G.m+G.m+i] = i
        Vs[G.m+G.m+i] = sqrt(d[i])
    end
    return sparse(Is, Js, Vs, G.n, G.n)
end


function getSparseIpLpDs_CW_weighted(G, X, Y, W, C)
    """
        avoid compute S, to get Ds with out memery
    """
    T = time()
    n, _ = size(X)
    d = ones(n)
    for (ID, u, v, w) in G.E
        d[u] += w
        d[v] += w
    end


    Ds = zeros(n)
    w = (C * W) / (2 * n)
    v = ones(n)

    temp1 = Y * v
    M_XYv = X * temp1

    temp2 = X' * v
    M_YXv = Y' * temp2
    Ms = M_XYv + M_YXv

    for i in 1:n
        Ds[i] = w * Ms[i]  
    end
    Is = zeros(Int32, G.m * 2 + G.n)
    Js = zeros(Int32, G.m * 2 + G.n)
    Vs = zeros(G.m * 2 + G.n)
    for (ID, u, v, w) in G.E
        Is[ID] = u
        Js[ID] = v
        Vs[ID] = -w
        Is[ID+G.m] = v
        Js[ID+G.m] = u
        Vs[ID+G.m] = -w
    end
    for i = 1:G.n
        Is[G.m+G.m+i] = i
        Js[G.m+G.m+i] = i
        Vs[G.m+G.m+i] = d[i] + Ds[i]
    end
    return sparse(Is, Js, Vs, G.n, G.n)
end


function getSparseIpLpDs_weighted(G, X, Y, alpha)
    """
        avoid compute S, to get Ds with out memery
    """

    T = time()
    _, n = size(X)
    d = zeros(n)
    for (ID, u, v, w) in G.E
        d[u] += w
        d[v] += w
    end
    Do = getSparse_D_squre_root(G)
    Ds = zeros(n)
    v = ones(n)
    temp1 = Y * Do * v
    M_XYv = Do * X' * temp1
    temp2 = X * Do * v
    M_YXv = Do * Y' * temp2
    Ms = M_XYv + M_YXv
    println("MS: ", sum(Ms))

    for i in 1:n
        Ds[i] = (alpha / 4) * Ms[i]
    end

    # println("Ds:", sum(Ds))
    Is = zeros(Int32, G.m * 2 + G.n)
    Js = zeros(Int32, G.m * 2 + G.n)
    Vs = zeros(G.m * 2 + G.n)
    for (ID, u, v, w) in G.E
        Is[ID] = u
        Js[ID] = v
        Vs[ID] = -w
        Is[ID+G.m] = v
        Js[ID+G.m] = u
        Vs[ID+G.m] = -w
    end
    for i = 1:G.n
        Is[G.m+G.m+i] = i
        Js[G.m+G.m+i] = i
        Vs[G.m+G.m+i] = d[i] + Ds[i] + 1
    end
    return sparse(Is, Js, Vs, G.n, G.n)
end

function getSparseIpLpDs(G, X, Y)
    """
        avoid compute S, to get Ds with out memery
    """
    T = time()
    _, n = size(X)
    #get Ds vector
    Ds = zeros(n)
    v = ones(n)

    temp1 = Y * v
    M_XYv = X' * temp1

    temp2 = X * v
    M_YXv = Y' * temp2
    Ms = M_XYv + M_YXv
    println("MS: ", sum(Ms))

    for i in 1:n
        Ds[i] = Ms[i]
    end


    d = ones(n)
    for (ID, u, v, w) in G.E
        d[u] += w
        d[v] += w
    end
    Is = zeros(Int32, G.m * 2 + G.n)
    Js = zeros(Int32, G.m * 2 + G.n)
    Vs = zeros(G.m * 2 + G.n)
    for (ID, u, v, w) in G.E
        Is[ID] = u
        Js[ID] = v
        Vs[ID] = -w
        Is[ID+G.m] = v
        Js[ID+G.m] = u
        Vs[ID+G.m] = -w
    end
    for i = 1:G.n
        Is[G.m+G.m+i] = i
        Js[G.m+G.m+i] = i
        Vs[G.m+G.m+i] = d[i] + Ds[i]
    end
    return sparse(Is, Js, Vs, G.n, G.n)
end



function getSparseIpL(G)
    d = ones(G.n)
    for (ID, u, v, w) in G.E
        d[u] += w
        d[v] += w
    end
    Is = zeros(Int32, G.m * 2 + G.n)
    Js = zeros(Int32, G.m * 2 + G.n)
    Vs = zeros(G.m * 2 + G.n)
    for (ID, u, v, w) in G.E
        Is[ID] = u
        Js[ID] = v
        Vs[ID] = -w
        Is[ID+G.m] = v
        Js[ID+G.m] = u
        Vs[ID+G.m] = -w
    end
    for i = 1:G.n
        Is[G.m+G.m+i] = i
        Js[G.m+G.m+i] = i
        Vs[G.m+G.m+i] = d[i]
    end
    return sparse(Is, Js, Vs, G.n, G.n)
end

function getSparseL(G)
    d = zeros(G.n) 
    for (ID, u, v, w) in G.E
        d[u] += w
        d[v] += w
    end
    Is = zeros(Int32, G.m * 2 + G.n)
    Js = zeros(Int32, G.m * 2 + G.n)
    Vs = zeros(G.m * 2 + G.n)
    for (ID, u, v, w) in G.E
        Is[ID] = u
        Js[ID] = v
        Vs[ID] = -w
        Is[ID+G.m] = v
        Js[ID+G.m] = u
        Vs[ID+G.m] = -w
    end
    for i = 1:G.n
        Is[G.m+G.m+i] = i
        Js[G.m+G.m+i] = i
        Vs[G.m+G.m+i] = d[i]
    end
    return sparse(Is, Js, Vs, G.n, G.n)
end

function get_random_M(d, n; type="col")
  

    a = Random.rand(Distributions.Uniform(0, 1), 10000)
    global M = zeros(d, n)
    if type == "col"
   
        for i = 1:n
            xi = Distributions.sample(a, d)
            sum_xi = sum(xi)
            for j = 1:d
                if sum_xi == 0
                    global M[j, i] = xi[j]
                else
                    M[j, i] = xi[j] / sum_xi
                end
            end
        end
    else
        for i = 1:d
            xi = Distributions.sample(a, n)
            sum_xi = sum(xi)
            for j = 1:n
                if sum_xi == 0
                    global M[i, j] = xi[j]
                else
                    global M[i, j] = xi[j] / sum_xi
                end

            end
        end
    end
    return M
end

function get_X(d, n, sparsity; distribution="normal")

    Random.seed!(10)
    o = Int(100000 * sparsity)
    rns = Int(100000 - o)
    if distribution == "normal"
        a = Random.rand(Normal(0.1, 0.1), rns)
        a = abs.(a)
        a = map((x) -> min(1, x), a)
        b = zeros(o)
        c = append!(a, b)
        rc = shuffle(c)
    else
        rootpath = "datasets_50k/"
        df_X = CSV.read(rootpath * "Twitter_X.csv", DataFrame)
        X_ = Matrix(df_X)
        interests = vec(X_)
        rc = interests
    end
    X = zeros(n, d)
    for i = 1:n
        xi = Distributions.sample(rc, d)
        if maximum(xi) == 0

            while maximum(xi) == 0
                xi = Distributions.sample(rc, d)
            end
        end
        X[i, :] = xi / sum(xi)
    end
    return X
end

function get_Y(d, n, sparsity; distribution="normal")

    Random.seed!(10)
    o = Int(100000 * sparsity)
    rns = Int(100000 - o)
    if distribution == "normal"
        a = Random.rand(Normal(0.1, 0.1), rns)
        a = abs.(a)
        a = map((x) -> min(1, x), a)
        b = zeros(o)
        c = append!(a, b)
        rc = shuffle(c)
    else
        rootpath = "datasets_50k/"
        df_Y = CSV.read(rootpath * "Twitter_Y.csv", DataFrame)
        Y_ = Matrix(df_Y)
        influence = vec(Y_)
        rc = influence
    end
    Y = zeros(d, n)
    for i = 1:d
        yi = Distributions.sample(rc, n)
        if maximum(yi) == 0

            while maximum(yi) == 0
                yi = Distributions.sample(rc, n)
            end
        end
        Y[i, :] = yi / sum(yi)
    end
    return Y
end

#
function powerLawForXY(n; alp=1.5, xmin=1, threshold)
    Random.seed!(round(Int, time() * 10000))
    x = rand(n)
    for i = 1:n
        x[i] = xmin * ((1 - x[i])^(-1.0 / (alp - 1.0)))
    end
    xm = x[argmax(x)]
    x ./= xm


    @inline function reduce(x)
        if x < threshold
            return 0
        else
            return x
        end
    end

    x = map(reduce, x)

    return x
end


# generate synthetic user-interest matrix
function get_X_powerlaw(k, n, alp, xmin, threshold)
    X = zeros(n, k)
    for i = 1:n
        xi = powerLawForXY(k; alp=alp, xmin=xmin, threshold=threshold) # for each row sample from powerlaw distribution
        if maximum(xi) == 0
            while maximum(xi) == 0
                xi = powerLawForXY(k; alp=alp, xmin=xmin, threshold=threshold)
            end
        end
        X[i, :] = xi / sum(xi)
    end
    return X
end

#generate synthetic topic-influence matrix.
function get_Y_powerlaw(k, n, alp, xmin, threshold)
    Y = zeros(k, n)
    for i = 1:k
        yi = powerLawForXY(n; alp=alp, xmin=xmin, threshold=threshold)
        if maximum(yi) == 0
            while maximum(xi) == 0
                yi = powerLawForXY(n; alp=alp, xmin=xmin, threshold=threshold)
            end
        end
        Y[i, :] = yi / sum(yi)
    end
    return Y
end

function get_Y_influencers_partition(s, n, k, c, weights, p, distribution)
    a, b = -1, 1
    h = (b - a) / (c)
    chunks = collect(a:h:b)
    chunk_keys = [i for i in 1:c]
    chunk_dict = Dict(k => Int[] for k in 1:c)
    for i in 1:n
        global chunk_key = -1
        for j in 1:c
            if s[i] >= chunks[j] && s[i] <= chunks[j+1]
                global chunk_key = j
                break
            end
        end
        append!(chunk_dict[chunk_key], i)
    end

    Y = zeros(k, n)
    n_influencers = floor(Int, n * p)
    for i = 1:k
        topic_type = StatsBase.sample(chunk_keys, Weights(weights))
        indices = StatsBase.sample(chunk_dict[topic_type], n_influencers, replace=true)

        r = 1
        for j in indices
            if distribution == "powerlaw"
                scores = powerLaw(n_influencers; alp=2.5, xmin=1) 
            elseif distribution == "expoential"
                scores = Exponential(n_influencers; lmd=1, xmin=1)
            elseif distribution == "uniform"
                scores = Uniform(n_influencers)
            end
            scores = scores ./ sum(scores)
            Y[i, j] = scores[r] 
            r += 1
        end
    end
    Y = Y ./ sum(Y, dims=2)
    return Y
end

