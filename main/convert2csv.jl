using JLD
using CSV,DelimitedFiles

function jld_to_csv(input_file::String, output_file::String)
    # Load data from JLD file
    data = load(input_file)["data"]    
    # Write data to CSV file
    open(output_file, "w") do io
        writedlm(io, data, ',')
    end
end
jld_file, output_file= ARGS
# Usage example:

jld_to_csv(jld_file,output_file)
println(jld_file," is converted to ", output_file)