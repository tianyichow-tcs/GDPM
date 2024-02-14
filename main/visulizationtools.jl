using CSV, Tables, DataFrames, Plots, StatsBase, JLD, StatsPlots, Printf, LaTeXStrings, Polynomials
include("Tools.jl")
include("Graph.jl")
include("Tools.jl")
include("Algorithm.jl")
include("ConvexOpt.jl")


function visualize_opinions(opinions, titles)
    f = 10
    l = @layout [
        grid(2, 2)
    ]
    fig = plot(
        opinions, layout=l, legend=false, seriestype=[histogram],
        title=[t for j in 1:1, t in titles],
        color="black", bins=-1:0.01:1, dpi=300, xtickfont=font(f), ytickfont=font(f), legendfont=font(f), guidefont=font(f), xlabel="opinions", ylabel="Number of users",
        #histogram(s_uni,xlabel="opinions",ylabel="Number of users",label=false,title=title,color="black",bins = -1:0.02:1,dpi=300,xtickfont=font(18),ytickfont=font(18),legendfont=font(18),guidefont=font(25),size=(800,600))
    )
    display(fig)
    savefig(fig, "paper_figures/opinions.pdf")
end


font_size = 16
function visualize_learning_rate(folder, filename, original, cw, C, Ts, max_ites, dataset)
    eps = [0.05, 0.1, 0.15, 0.2]
    Ls = [10, 100, 1000, 10000]
    #Ts = [5000,10000,15000,20000]
    p_o, pd_o = original
    p_cw, pd_cw = cw
    # p_list=[p_o,p_cw]
    # pd_list = [pd_o,pd_cw]
    T = []
    for epsilon in eps
        data = []
        for i in 1:4
            #read opt curves 
            L = Ls[i]
            T = Ts[i]
            p_opt_list_name = "Opts/" * folder * "/opts_p_" * filename * "_L" * string(L) * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
            pd_opt_list_name = "Opts/" * folder * "/opts_pd_" * filename * "_L" * string(L) * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
            p_opt_list = load(p_opt_list_name)["data"]
            pd_opt_list = load(pd_opt_list_name)["data"]
            reduction = 100 * ((pd_opt_list) / pd_cw)
            # gap = max_ites[i] - length(reduction)
            # result = vcat(reduction,last(reduction)*ones(Int(gap)))
            push!(data, reduction[1:max_ites[i]])
        end
        #reduction_cw = 100*((pd_cw)/pd_o)
        #reduction_data = (pd_o*ones(4,T)-data)/pd_o
        max_lim = max_ites[4] + 500
        fig = plot(data, label=[L"$L$=10" L"$L$=100" L"$L$=1000" L"$L$=10000"], lw=3, xlabel="Iterations", ylabel=L"$I(G)$ reduction rate(%)", legend=:topright,
            xtickfont=font(font_size),
            ytickfont=font(font_size),
            guidefont=font(font_size),
            legendfont=font(font_size), xlim=[0, max_lim])
        savefig(fig, "paper_figures/fig_exp_" * filename * "_learningrates_eps" * string(epsilon) * ".pdf")
        display(fig)
    end
end

function visualize_budgets(folder, filename, original, cw, C, Ts, max_ites, dataset)
    epsilons = [0.05, 0.1, 0.15, 0.2]
    Ls = [10, 100, 1000, 10000]
    #Ts = [5000, 10000, 15000, 20000]
    p_o, pd_o = original
    p_cw, pd_cw = cw
    # p_list=[p_o,p_cw]
    # pd_list = [pd_o,pd_cw]
    for i in 1:4
        data = []
        for epsilon in epsilons
            #read opt curves 
            L = Ls[i]
            T = Ts[i]
            p_opt_list_name = "Opts/" * folder * "/opts_p_" * filename * "_L" * string(L) * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
            pd_opt_list_name = "Opts/" * folder * "/opts_pd_" * filename * "_L" * string(L) * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
            p_opt_list = load(p_opt_list_name)["data"]
            pd_opt_list = load(pd_opt_list_name)["data"]
            reduction = 100 * ((pd_opt_list) / pd_cw)
            push!(data, reduction[1:max_ites[i]])

        end
        #reduction_cw = 100*((pd_cw)/pd_o)
        max_lim = max_ites[i] + 200
        fig = plot(data, label=[L"$θ=0.05$" L"$θ=0.1$" L"$θ=0.15$" L"$θ=0.2$"], lw=3, xlabel="Iterations", ylabel=L"$I(G)$ reduction rate(%)", legend=:topright,
            xtickfont=font(font_size),
            ytickfont=font(font_size),
            guidefont=font(font_size),
            legendfont=font(font_size), xlim=[0, max_lim])
        #scatter!([0],[pd_o],label=L"$G_{o}$",ms=3,mc=:red)
        #display(scatter!([0],[reduction_cw],label=L"$G_{cw}$",ms=3,mc=:red,dpi=300))
        savefig(fig, "paper_figures/fig_exp_" * filename * "_budgets_L" * string(Ls[i]) * ".pdf")
        display(fig)
    end
end

function visualize_C(folder, filename, Ts, max_ites, dataset)
    Cs = [0.1, 0.2, 0.3, 0.4]
    epsilon = 0.1
    L = 10

    #Ts = [5000, 10000, 15000, 20000]
    # p_o,pd_o =original 
    # p_cw,pd_cw = cw
    # p_list=[p_o,p_cw]
    # pd_list = [pd_o,pd_cw]
    for i in 1:4
        data = []
        T = Ts[i]
        max_lim = max_ites[4] + 500
        for C in Cs
            p_opt_list_name = "Opts/" * folder * "/opts_p_" * filename * "_L" * string(L) * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
            pd_opt_list_name = "Opts/" * folder * "/opts_pd_" * filename * "_L" * string(L) * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
            p_opt_list = load(p_opt_list_name)["data"]
            pd_opt_list = load(pd_opt_list_name)["data"]
            reduction = 100 * ((pd_opt_list) / pd_opt_list[1])
            push!(data, reduction[1:max_ites[i]])

        end
        fig = plot(data, label=[L"$C=0.1$" L"$C=0.2$" L"$C=0.3$" L"$C=0.4$"], lw=3, xlabel="Iterations", ylabel=L"$I(G)$ reduction rate(%)", legend=:topright,
            xtickfont=font(font_size),
            ytickfont=font(font_size),
            guidefont=font(font_size),
            legendfont=font(font_size), xlim=[0, max_lim])
        #scatter!([0],[pd_o],label=L"$G_{o}$",ms=3,mc=:red)
        #display(scatter!([0],[reduction_cw],label=L"$G_{cw}$",ms=3,mc=:red,dpi=300))
        savefig(fig, "paper_figures/fig_exp_" * filename * "_weights_C" * string(Cs[i]) * ".pdf")
        display(fig)
    end
end


function visualize_C_group(path, filename, baseline_name)
    L = 10
    epsilon = 0.1

    Cs = [0, 0.1, 0.2, 0.3, 0.4]
    baseline_bridge = []
    baseline_neutral = []
    # baseline = vec([0.10806422190968422, 0.11569853578882319, 0.1203655346305082,0.12343330551920226])

    for i in 2:5
        T = 100
        C = Cs[i]
        pd_opt_list_name_b = "Opts/baseline/bridge/opts_pd_" * baseline_name * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
        pd_opt_list_b = load(pd_opt_list_name_b)["data"]
        pd_opt_list_name_n = "Opts/baseline/neutral/opts_pd_" * baseline_name * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
        pd_opt_list_b = load(pd_opt_list_name_b)["data"]
        reduction_b = (minimum(pd_opt_list_b)) / pd_opt_list_b[1]

        pd_opt_list_n = load(pd_opt_list_name_n)["data"]
        reduction_n = (minimum(pd_opt_list_n)) / pd_opt_list_n[1]
        append!(baseline_bridge, reduction_b)
        append!(baseline_neutral, reduction_n)
        println("C=", C, " pd_cw_b=", pd_opt_list_b[1], " pd_cw_n=", pd_opt_list_n[1])
    end



    data = zeros(5, 3)
    data[1, :] = ones(3)
    for i = 2:5
        pd_idx = []
        C = Cs[i]

        #read opt curves 
        T = 510
        #p_opt_list_name = "Opts/"*path*"/opts_p_"*filename*"_L" * string(L) * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
        pd_opt_list_name = "Opts/" * path * "/opts_pd_" * filename * "_L" * string(L) * "_eps" * string(epsilon) * "_C" * string(C) * "_T" * string(T) * ".jld"
        #p_opt_list = load(p_opt_list_name)["data"]
        pd_opt_list = load(pd_opt_list_name)["data"]
        reduction = (minimum(pd_opt_list)) / pd_opt_list[1]
        #reduction = (cw[2]-pd_opt_list[end-300])/cw[2]
        data[i, 1] = reduction

        data[i, 2] = baseline_neutral[i-1]
        data[i, 3] = baseline_bridge[i-1]
        println("C=", C, " pd_cw=", pd_opt_list[1])

    end
    return data, reduction_cw
end


function algorithm_behavior_simulation(filename, epsilon, L, T, C, s_distribution, inf_distribution, c, weights)
    networkType = "unweighted"

    rootpath = "/Users/tianyi/Desktop/research/CodeBase/ConvexOpitimization/data/"
    fileName = string(rootpath, filename, ".txt")
    G0 = readGraphs(fileName, networkType)
    G = getLLC(G0)
    n = G.n
    k = 100


    global s = []

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



    n = G.n
    k = 100
    p = 0.02                    # percent of influencers 
    # c = 3                       # 3 chanks of inate opinion spectrum
    # weights = [0.3, 0.4, 0.3]   # weights of 

    # get interest and influence matrix
    X = get_X_powerlaw(k, n, 2.5, 2, 0.25)
    Y = get_Y_influencers_partition(s, n, k, c, weights, p, inf_distribution)



    W = length(G.E)


    s = s .- (s' * ones(n)) / n
    _, z_o, p, pd = Approximation_PD_index_CW_weigted(G, X, Y, s, W, C; eps=1e-6)
    # X_T = 
    X_T, p_opt_list, pd_opt_list, errors = GradientDecent(G, s, X, Y, C, W, n, k, T, L, epsilon)
    println("pd reduction:", last(pd_opt_list) / pd_opt_list[1])
    _, z_t, p, pd = Approximation_PD_index_CW_weigted(G, X_T, Y, s, W, C; eps=1e-6)

    #change of x
    x_diff = X_T' * ones(n) - X' * ones(n)


    T_os = Y * s
    T_oz = Y * z_o
    T_tz = Y * z_t
    return x_diff, T_os, T_oz, T_tz

end

function plot_alg_behavior_simulation(x, y, titles, figurename)
    #title("Julia Plots Like a Boss")
    scatter(label=false)
    for i in 1:7
        scatter!(x, y, label=false)
    end

    display(scatter!(x, y, label=false, color="grey0",
        ylabel="Topic changes δ", xlabel="Weighted average opinion τ in topics", title=titles,
        xtickfont=font(12),
        ytickfont=font(12),
        titlefont=font(11),
        guidefont=font(12),
        legendfont=font(8)))
    savefig(figurename)
end
