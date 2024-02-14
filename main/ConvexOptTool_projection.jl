#in this file we buit the convex optimization algorithm
include("Algorithm.jl")


@inline function PBox_lu(x, l, u)
    x_box = map(max, x, l)
    x_box = map(min, x_box, u)
    return x_box
end

#projection of the matrix elements to feasible range.
function orthogonalProjectionBoxIntersectionHyperplane(y, l, u, k)
    T = []
    @inline function get_T(y, l, u)
        append!(T, y - l, y - u)
    end
    map(get_T, y, l, u)

    t_L = -Inf
    t_U = Inf

    @inline function getX(t, y, l, u, k)
        x_box = map(max, y - t * ones(k), l)
        x_box = map(min, x_box, u)
        return x_box
    end

    @inline function g(t, y, l, u, k)
        return sum(getX(t, y, l, u, k))
    end

    t = Inf
    value = Inf
    while value != 1
        t = T[rand(1:length(T))] 
        value = g(t, y, l, u, k)
        if value == 1
            break
        elseif value > 1
            t_L = t
            T = T[T.>t]

        elseif value < 1
            t_U = t
            T = T[T.<t]

        end

        if length(T) == 0

            t = t_L - (g(t_L, y, l, u, k) - 1) * (t_U - t_L) / (g(t_U, y, l, u, k) - g(t_L, y, l, u, k))
            break
        end
    end

    value = g(t, y, l, u, k)

    x = getX(t, y, l, u, k)

    return x
end

function ComputePBox_lu(x, l, u, k)
    return orthogonalProjectionBoxIntersectionHyperplane(x, l, u, k)
end


# compute lower bound and uper bound of user-intesest matrix
function Get_X_L_U(X, epsilon)

    n, d = size(X)
    X_eps = epsilon * ones(n, d)

    X_l = X - X_eps
    X_l = map(x -> max(0.0, x), X_l)
    X_u = X + X_eps
    X_u = map(x -> min(1.0, x), X_u)
    return X_l, X_u
end

# used to mask certain topics, for example, don't change political contents
function Get_X_L_U_masked(X, epsilon, label)

    n, k = size(X)
    X_eps = epsilon * ones(n, k)

    X_l = X - X_eps
    X_l = map(x -> max(0.0, x), X_l)
    X_u = X + X_eps
    X_u = map(x -> min(1.0, x), X_u)

    for i = 1:k
        if label[i] == 1

            X_u[:, i] = X[:, i]
        end
    end
    return X_l, X_u
end

# compute the gradient
function Gradient(G, s, X_T, Y, C, W, n, k)
    one = ones(k)
    _, z, p, pd = Approximation_PD_index_CW_weigted(G, X_T, Y, s, W, C; eps=1e-6)
    w = (C * W) / (2.0 * n)
    gradient = w * (2 * z * (z' * Y') - z .* z * one' - ones(n) * ((z' .* z') * Y'))
    return gradient, p, pd
end
