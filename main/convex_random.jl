using CSV, LaTeXStrings
using DataFrames
using Random
using SparseArrays
using Laplacians
using LinearAlgebra

using Convex
using SCS
using GLPK
using Plots
using StatsPlots
using JLD

include("Graph.jl")
include("Tools.jl")
include("Algorithm.jl")
include("ConvexOpt.jl")
include("Baseline_onestep.jl")
C = 0.1

networkType = "unweighted"


filepath = ARGS[1]
V = parse(Int64, ARGS[2])
k = parse(Int64, ARGS[3])
rootpath = "data/"*filepath*"/" 


filename = filepath
fileName = string(rootpath, filename, ".txt")

theta = 0.1   # budget

println("Start processing:", filename)
G0 = readGraphs(fileName, networkType)
G = getLLC(G0)
n = G.n

path = rootpath
s = load(path*"s.jld")["data"]
X = load(path*"X.jld")["data"]
Y = load(path*"Y.jld")["data"]

W = length(G.E)

v = ones(n)
s_mean = s - ((s' * v) / n) * v

X_l, X_u = Get_X_L_U(X, theta)

t1 = time()


IpL =getIpL(G)
X = Variable(V,k)
w = (C * W) / (2 * n)
A = (IpL + Diagonal(w*(X*Y+Y'*X')*ones(n)))


P = A - w * (X * Y + Y' * X')

p = minimize(matrixfrac(s_mean,P), X <= X_u, X>= X_l, X>=0, X*ones(k)==1)
solve!(p, SCS.Optimizer )

T = time()-t1
println("Optimal solution:", p.optval)
println("Optimization time:", T)

println("check the constraint of X:")
println(round.(evaluate(X*ones(k))))
