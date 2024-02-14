using CSV, LaTeXStrings
using DataFrames
using Random
using SparseArrays
using Laplacians
using Plots
using StatsPlots
using JLD

include("Graph.jl")
include("Tools.jl")
include("Algorithm.jl")
include("ConvexOpt.jl")
include("Baseline_onestep.jl")

"""
    In this experiment, we run simulation experiment on algorithm 3, baseline 1 and 2.
    Sythetic opinion is fixed as polarized distribution and sythetic interest and influence are from powerlaw distribution.

    Input: G, ϵ, L, C, T_max; 

"""

println("We gonna assign parameters as follow ing: \n",
    "G=", ARGS[1], " ",
    "theta=", ARGS[2], " ",
    "L=", ARGS[3], " ",
    "T=", ARGS[4], " ",
    "s=", ARGS[5], " ",
    "influencer distribution=", ARGS[6]
)
C = 0.1

networkType = "unweighted"
rootpath = "../graph_data/" # input path

outputpath = "../output/"  # out put path

s_distributions = ["uniform", "powerlaw", "expoential", "polarized"]

filename = ARGS[1]
fileName = string(rootpath, filename, ".txt")

theta = parse(Float64, ARGS[2])   # budget
L = parse(Int64, ARGS[3])           # learing rate
T = parse(Int64, ARGS[4])           # Iterations of GradientDecent
s_distribution = ARGS[5]            # distribution for generating innate opinions
inf_distribution = ARGS[6]          # distributions for generating influencers


lg = open(outputpath*"Logs/data/log_convexopt_simulation_" * filename * ".txt", "a")
println("Start processing:", filename)
G0 = readGraphs(fileName, networkType)
G = getLLC(G0)
n = G.n

if s_distribution == "uniform"
    s = Uniform(n)
elseif s_distribution == "powerlaw"
    s = powerLaw(n)
elseif s_distribution == "expoential"
    s = Exponential(n)
else
    s = Polarized(n)
end

s = s * 2 - ones(n)  # normalize to [-1,1]

# read X,y
n = G.n
k = 100
p = 0.02                    # percent of influencers 
c = 3                       # 3 chanks of inate opinion spectrum
weights = [0.3, 0.4, 0.3]   # weights of 

# get interest and influence matrix
X = sparse(get_X_powerlaw(k, n, 2.5, 2, 0.25))
Y = sparse(get_Y_influencers_partition(s, n, k, c, weights, p, inf_distribution))

x_name = outputpath * "XTs/data/convexopt/X_" * filename * "_L" * string(L) * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"
save(x_name, "data", X)

println(repeat("+", 200))
println("Graph:", fileName)
println("nodes: ", G.n, " \t", "edges: ", length(G.E), "\t m/n:", length(G.E) / G.n)

println(lg, repeat("+", 200))
println(lg, "Graph:", fileName)
println(lg, "nodes: ", G.n, " \t", "edges: ", length(G.E), "\t m/n:", length(G.E) / G.n)

s_notation = s_distribution
X_notation = "powerlaw"

W = length(G.E)


println("------------------------------Opinion:", s_notation, ",  X and Y:", X_notation, "--------------------------------------------------------------------------------------------------")
println(lg, "------------------------------Opinion:", s_notation, ",  X and Y:", X_notation, "--------------------------------------------------------------------------------------------------")
#original opinions
_, p_o, pd_o = Approx(G, s)
T_approx, az, p_cw, pd_cw = Approximation_PD_index_CW_weigted(G, X, Y, s, W, C; eps=1e-6)

pd_reduce = pd_cw / pd_o
p_reduce = p_cw / p_o
println("Polarization_original:", p_o, "\t PD_original:", pd_o, "\t polarization_with_recommendation:", p_cw, "\t PD_with_recommendation:", pd_cw)
println("Polarization reduction:", p_reduce, "\t PD index reduction:", pd_reduce)
println("Approximation_running_time:", T_approx)
println("-----------------------------------------------------------------end---------------------------------------------------------------------------------------------")
println(lg, "Polarization_original:", p_o, "\t PD_original:", pd_o, "\t polarization_with_recommendation:", p_cw, "\t PD_with_recommendation:", pd_cw)
println(lg, "Polarization reduction:", p_reduce, "\t PD index reduction:", pd_reduce)
println(lg, "Approximation_running_time:", T_approx)
println(lg, "-----------------------------------------------------------------end---------------------------------------------------------------------------------------------")

## Convex optimization
println("----------------------------------Start Optimization, L= ", L, " theta=", theta, " Max T=", T, "----------------------------------")
println(lg, "----------------------------------Start Optimization, L= ", L, " theta=", theta, "Max T=", T, "----------------------------------")

stat_time = time()
X_T, p_opt_list, pd_opt_list, errors, T_opt = GradientDecent(G, s, X, Y, C, W, n, k, T, L, theta)


error_name = outputpath * "Errors/data/convexopt/errors_" * filename * "_L" * string(L) * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"
xt_name = outputpath * "XTs/data/convexopt/XTs_" * filename * "_L" * string(L) * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

pd_opt_list_name = outputpath * "Opts/data/convexopt/opts_pd_" * filename * "_L" * string(L) * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"
p_opt_list_name = outputpath * "Opts/data/convexopt/opts_p_" * filename * "_L" * string(L) * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"
save(error_name, "data", errors)
save(xt_name, "data", X_T)
save(pd_opt_list_name, "data", pd_opt_list)
save(p_opt_list_name, "data", p_opt_list)

op_pd_reduce = minimum(pd_opt_list) / pd_cw
op_p_reduce = minimum(p_opt_list) / p_cw
println("op_p_reduce: ", op_p_reduce, "\t", "op_pd_reduce: ", op_pd_reduce)
println(lg, "op_p_reduce: ", op_p_reduce, "\t", "op_pd_reduce: ", op_pd_reduce)

end_time = time()
total_time_opt = end_time - stat_time
println("Time for optimization:", total_time_opt, " Iterations:", T_opt)
println(lg, "Time for optimization:", total_time_opt, " Iterations:", T_opt)

fig = plot(100 * ((pd_opt_list) / pd_opt_list[1]), title="PD index optimization", xlabel="Iteration", ylabel=L"$I(G)$ reduction rate(%)", label="L=" * string(L) * ", theta=" * string(theta), lw=3)
#plot(1:T, [hcat(p_opt_list) hcat(pd_opt_list)], title="PD index optimization", label=["P index" "PD index"], lw=3)
figname = outputpath * "figs/data/convexopt/" * filename * "_pd_opt_list_L" * string(L) * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".png"
savefig(fig, figname)
println("----------------------------------Saving figs, end optimization----------------------------------------------\n")
println(lg, "----------------------------------Saving figs, end optimization----------------------------------------------\n")

println(repeat("+", 200))
println(lg, repeat("+", 200))

T = 100
## baseline neutral
method = "neutral"
println("----------------------------------Baseline: neutral----------------------------------------------\n")
println(lg, "----------------------------------Baseline: neutral----------------------------------------------\n")

X = load(x_name)["data"]
start_time = time()

X_T_b2, p_b2_list, pd_b2_list, errors_b2, T_b2 = Baseline_Neutral(G, X, Y, n, k, theta, T, C)

end_time = time()
total_time_b2 = end_time - start_time

error_b2_name = outputpath * "Errors/data/baseline/neutral_errors_" * filename * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

xt_b2_name = outputpath * "XTs/data/baseline/XTs_neutral_" * filename * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

pd_b2_list_name = outputpath * "Opts/data/baseline/neutral_pd_" * filename * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

p_b2_list_name = outputpath * "Opts/data/baseline/neutral_p_" * filename * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

save(error_b2_name, "data", errors_b2)
save(pd_b2_list_name, "data", pd_b2_list)
save(p_b2_list_name, "data", p_b2_list)
save(xt_b2_name, "data", X_T_b2)


b2_pd_reduce = minimum(pd_b2_list) / pd_cw
b2_p_reduce = minimum(p_b2_list) / p_cw
println("b2_p_reduce: ", b2_p_reduce, "\t", "b2_pd_reduce: ", b2_pd_reduce)
println(lg, "b2_p_reduce: ", b2_p_reduce, "\t", "b2_pd_reduce: ", b2_pd_reduce)

println("Time for optimization:", total_time_b2, " Iterations:", T_b2)
println(lg, "Time for optimization:", total_time_b2, " Iterations:", T_b2)


fig = plot(100 * (pd_b2_list / pd_b2_list[1]), title="PD index optimization", xlabel="Iteration", ylabel="PD index", label="T=" * string(T) * ", theta=" * string(theta), lw=3)

figname = outputpath * "figs/data/baseline/neutral_" * filename * "_pd" * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".png"
savefig(fig, figname)
println("----------------------------------Saving figs, end optimization----------------------------------------------\n")
println(lg, "----------------------------------Saving figs, end optimization----------------------------------------------\n")

close(lg)

T = 100
lg = open(outputpath*"Logs/data/log_convexopt_simulation_" * filename * ".txt", "a")
# start baseline
method = "bridge"
println("----------------------------------Baseline: bridge----------------------------------------------\n")
println(lg, "----------------------------------Baseline: bridge----------------------------------------------\n")

X = load(x_name)["data"]
start_time = time()

X_T_b1, p_b1_list, pd_b1_list, errors_b1, T_b1 = Baseline_Bridge(G, X, Y, n, k, theta, T, C)

end_time = time()
total_time = end_time - start_time

error_b1_name = outputpath * "Errors/data/baseline/bridge_errors_" * filename * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

xt_b1_name = outputpath * "XTs/data/baseline/XTs_bridge_" * filename * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

pd_b1_list_name = outputpath * "Opts/data/baseline/bridge_pd_" * filename * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

p_b1_list_name = outputpath * "Opts/data/baseline/bridge_p_" * filename * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".jld"

save(error_b1_name, "data", errors_b1)
save(pd_b1_list_name, "data", pd_b1_list)
save(p_b1_list_name, "data", p_b1_list)
save(xt_b1_name, "data", X_T_b1)


b1_pd_reduce = minimum(pd_b1_list) / pd_cw
b1_p_reduce = minimum(p_b1_list) / p_cw
println("b1_p_reduce: ", b1_p_reduce, "\t", "b1_pd_reduce: ", b1_pd_reduce)
println(lg, "b1_p_reduce: ", b1_p_reduce, "\t", "b1_pd_reduce: ", b1_pd_reduce)

println("Time for optimization:", total_time, " Iterations:", T_b1)
println(lg, "Time for optimization:", total_time, " Iterations:", T_b1)


fig = plot(100 * (pd_b1_list / pd_b1_list[1]), title="PD index optimization", xlabel="Iteration", ylabel="PD index", label="T=" * string(T) * ", theta=" * string(theta), lw=3)

figname = outputpath * "figs/data/baseline/bridge_" * filename * "_pd" * "_eps" * string(theta) * "_C" * string(C) * "_T" * string(T) * ".png"
savefig(fig, figname)
println("----------------------------------Saving figs, end optimization----------------------------------------------\n")
println(lg, "----------------------------------Saving figs, end optimization----------------------------------------------\n")


close(lg)

lg_table = open(outputpath*"Logs/data/log_syn_exp.txt", "a")
println(lg_table, filename,"\t", round(op_pd_reduce*100,digits=4),"\t", round(b2_pd_reduce*100,digits=4), "\t", round(b1_pd_reduce*100,digits=4), "\t", total_time_opt,"\t", total_time_b2, "\t", total_time)
close(lg_table)