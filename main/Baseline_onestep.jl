# Baseline Algorithms

include("Algorithm.jl")
include("ConvexOpt.jl")
using SparseArrays

function find_j(x, x_l, x_u, T_diff, k)
    @inline function less(x, x_u)
        if x < x_u
            return 1
        else
            return Inf
        end
    end
    @inline function great(x, x_l)
        if x > x_l
            return 1
        else
            return 0
        end
    end

    candidates_j = map(less, x, x_u)
    j = argmin(candidates_j .* T_diff)
    
    candidates_j_ = map(great, x, x_l)
    j_ = argmax(candidates_j_ .* T_diff)
    return j, j_

end

function Baseline_Neutral(G, X, Y, n, k, epsilon, T, C)
    X_l, X_u = Get_X_L_U(X, epsilon)
    global X_t = X
    global p_b2_list = []
    global pd_b2_list = []
    global errors = []
    global T_converge = T
    for t in 1:T
        _, z, p, pd = Approximation_PD_index_CW_weigted(G, X_t, Y, s, W, C; eps=1e-6)
        v_one = ones(n)
        z_avg = (z' * v_one) / n
        T_vec = Y * z
        T_diff = map(abs, (T_vec - z_avg * ones(k)))

        for i in 1:n
            x_l = X_l[i, :]
            x_u = X_u[i, :]
            j = 0
            j_ = 0
            delta = 1

            x = X_t[i, :]

            #
            j, j_ = find_j(x, x_l, x_u, T_diff, k)
            if j == j_
                continue
            end
            delta = min(X_u[i, j] - x[j], x[j_] - X_l[i, j_])
            x[j] = x[j] + delta
            x[j_] = x[j_] - delta

            global X_t[i, :] = x

        end

        error = norm(X_t * ones(k) - ones(n), 2)
        # println("error=", error)
        append!(errors, error)

        append!(p_b2_list, p)
        append!(pd_b2_list, pd)
    end
    return X_t, p_b2_list, pd_b2_list, errors, T_converge
end



function find_candicates(x, x_l, x_u, z, z_i, z_avg, T_opinions, T_diff, k)
    @inline function less(x, x_u)
        if x < x_u
            return 1
        else
            return 0
        end
    end
    @inline function greater(x, x_l)
        if x > x_l
            return 1
        else
            return 0
        end
    end
    candidates_j = map(less, x, x_u)
    masked_candidates_j = candidates_j .* T_opinions
    temp = -z_i * masked_candidates_j  
    j = argmax(temp)
    candidates_j_ = map(greater, x, x_l)
    masked_condidates_j_ = candidates_j_ .* T_diff  
    t = Inf
    j_ = -1
    for i in 1:k
        if masked_condidates_j_[i] < t && masked_condidates_j_[i] != 0 && z_i * T_opinions[i] > 0
            t = masked_condidates_j_[i]
            j_ = i
        end
    end
    if j_ == -1
        code = 0
    else
        code = 1
    end

    return code, j, j_
end

function Baseline_Bridge(G, X, Y, n, k, epsilon, T, C)
    X_l, X_u = Get_X_L_U(X, epsilon)
    global X_t = X
    global p_b1_list = []
    global pd_b1_list = []
    global errors = []

    global T_converge = T

    for t in 1:T
        _, z, p, pd = Approximation_PD_index_CW_weigted(G, X_t, Y, s, W, C; eps=1e-6)
        v_one = ones(n)
        z_avg = (z' * v_one) / n
        T_opinions = Y * z
        T_diff = map(abs, (T_opinions - z_avg * ones(k)))

        for i in 1:n
            x_l = X_l[i, :]
            x_u = X_u[i, :]
            z_i = z[i]
            j = 0
            j_ = 0
            delta = 1
            count = 0
            x = X_t[i, :]
            code, j, j_ = find_candicates(x, x_l, x_u, z, z_i, z_avg, T_opinions, T_diff, k)
            if j == j_ || code == 0
                continue
            end

            delta = min(X_u[i, j] - x[j], x[j_] - X_l[i, j_])
            x[j] = x[j] + delta
            x[j_] = x[j_] - delta

            global X_t[i, :] = x

        end

        error = norm(X_t * ones(k) - ones(n), 2)
        append!(errors, error)
        append!(p_b1_list, p)
        append!(pd_b1_list, pd)

    end

    return X_t, p_b1_list, pd_b1_list, errors, T_converge
end