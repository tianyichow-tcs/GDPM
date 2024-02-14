struct Graph
    n::Int128 # |V|
    m::Int128 # |E|
    V::Array{Int128,1} # V[i] = Real Index of node i
    E::Array{Tuple{Int128,Int128,Int128,Float64},1} # (ID, u, v, w) in Edge Set
end

function readGraphs(fileName, graphType) 
    n = 0
    origin = Dict{Int128,Int128}()
    label = Dict{Int128,Int128}()
    edge = Set{Tuple{Int128,Int128,Float64}}()

    getid(x::Int128) = haskey(label, x) ? label[x] : label[x] = n += 1

    open(fileName) do f1
        for line in eachline(f1)
            # Read origin data from file
            buf = split(line)
            u = parse(Int128, buf[1])
            v = parse(Int128, buf[2])
            if graphType == "weighted"
                w = parse(Float64, buf[3])
            else
                w = 1.0
            end
            if u == v
                continue
            end
            # Label the node
            u1 = getid(u)
            v1 = getid(v)
            origin[u1] = u
            origin[v1] = v
            # Store the edge
            if u1 > v1
                u1, v1 = v1, u1
            end
            push!(edge, (u1, v1, w))
        end
    end

    m = length(edge)
    V = Array{Int128,1}(undef, n)
    E = Array{Tuple{Int128,Int128,Int128,Float64},1}(undef, m)

    for i = 1:n
        V[i] = origin[i]
    end

    ID = 0
    for (u, v, w) in edge
        ID = ID + 1
        E[ID] = (ID, u, v, w)
    end

    return Graph(n, m, V, E)
end


function readGraph(fileName, fileName_newedges, graphType)
    n = 0
    origin = Dict{Int128,Int128}()
    label = Dict{Int128,Int128}()
    edge = Set{Tuple{Int128,Int128,Float64}}()

    getid(x::Int128) = haskey(label, x) ? label[x] : label[x] = n += 1

    open(fileName) do f1
        for line in eachline(f1)
          
            buf = split(line)
            u = parse(Int128, buf[1])
            v = parse(Int128, buf[2])
            if graphType == "weighted"
                w = parse(Float64, buf[3])
            else
                w = 1.0
            end
            if u == v
                continue
            end
      
            u1 = getid(u)
            v1 = getid(v)
            origin[u1] = u
            origin[v1] = v
          
            if u1 > v1
                u1, v1 = v1, u1
            end
            push!(edge, (u1, v1, w))
        end
    end

    if fileName_newedges != "None"
        open(fileName_newedges) do f1
            for line in eachline(f1)
               
                buf = split(line)
                u = parse(Int128, buf[1])
                v = parse(Int128, buf[2])
                if graphType == "weighted"
                    w = parse(Float64, buf[3])
                else
                    w = 1.0
                end
                if u == v
                    continue
                end
               
                u1 = getid(u)
                v1 = getid(v)
                origin[u1] = u
                origin[v1] = v
                
                if u1 > v1
                    u1, v1 = v1, u1
                end
                push!(edge, (u1, v1, w))
            end
        end
    end


    m = length(edge)
    V = Array{Int128,1}(undef, n)
    E = Array{Tuple{Int128,Int128,Int128,Float64},1}(undef, m)

    for i = 1:n
        V[i] = origin[i]
    end

    ID = 0
    for (u, v, w) in edge
        ID = ID + 1
        E[ID] = (ID, u, v, w)
    end

    return Graph(n, m, V, E)
end

function BFS(st, g) 
    q = []
    h = Array{Int8,1}(undef, size(g, 1))
    fill!(h, 0)
    push!(q, st)
    h[st] = 1
    front = 1
    rear = 1
    while front <= rear
        u = q[front]
        front = front + 1
        for v in g[u]
            if h[v] == 0
                h[v] = 1
                rear = rear + 1
                push!(q, v)
            end
        end
    end
    return q
end

function getLLC(G::Graph) 
    g = Array{Array{Int128,1},1}(undef, G.n)
    for i = 1:G.n
        g[i] = []
    end
    for (ID, u, v, w) in G.E
        push!(g[u], v)
        push!(g[v], u)
    end
    
    n2 = 0
    S = Array{Int128,1}(undef, G.n)
    visited = Array{Int8,1}(undef, G.n)
    fill!(visited, 0)
    for i = 1:G.n
        if visited[i] == 0
            nodeSet = BFS(i, g)
            for x in nodeSet
                visited[x] = 1
            end
            tn = size(nodeSet, 1)
            if (tn > n2)
                n2 = tn
                for j = 1:n2
                    S[j] = nodeSet[j]
                end
            end
        end
    end
   
    label = Array{Int128,1}(undef, G.n)
    fill!(label, 0)
    V2 = Array{Int128,1}(undef, n2)
    for i = 1:n2
        label[S[i]] = i
        V2[i] = G.V[S[i]]
    end
    E2 = []
    nID = 0
    for (ID, u, v, w) in G.E
        if (label[u] > 0) && (label[v] > 0)
            nID += 1
            push!(E2, (nID, label[u], label[v], w))
        end
    end
    m2 = size(E2, 1)
    return Graph(n2, m2, V2, E2)
end
