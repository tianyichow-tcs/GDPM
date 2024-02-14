
using Distributed

using SparseArrays

include("Algorithm.jl")
include("ConvexOptTool_projection.jl")



function GradientDecent(G, s, X, Y, C, W, n, k, T, L, epsilon)
    global X_l, X_u = Get_X_L_U(X, epsilon)
    global X_t = X  
    global GD_sum_t = zeros(n, k)
    global A_t = 0
    global p_opt_list = []
    global pd_opt_list = []
    global errors = []
    global T_converge = 0

    for t in 1:T
        t1 = time()
        GD_xt, p, pd = Gradient(G, s, X_t, Y, C, W, n, k) 
        append!(p_opt_list, p)
        append!(pd_opt_list, pd)
        V_temp = X_t - (1.0 / L) * (GD_xt)
        V_t = zeros(n, k)
        for i = 1:n
            v_i = ComputePBox_lu(V_temp[i, :], X_l[i, :], X_u[i, :], k)
            V_t[i, :] = v_i
        end

        alpha_t = (convert(Float64, t) + 1.0) / 2.0
        GD_sum_t += alpha_t * GD_xt 

        W_temp = X - (1.0 / convert(Float64, 2 * L)) * GD_sum_t
        W_t = zeros(n, k)
        for i = 1:n
            w_i = ComputePBox_lu(W_temp[i, :], X_l[i, :], X_u[i, :], k) 
            W_t[i, :] = w_i
        end
        A_t += alpha_t
        tau_t = alpha_t / A_t

        global X_t = tau_t * V_t + (1.0 - tau_t) * W_t
        t4 = time()
        error = norm(X_t * ones(k) - ones(n), 2)
        append!(errors, error)
    end


    return X_t, p_opt_list, pd_opt_list, errors, T_converge

end
