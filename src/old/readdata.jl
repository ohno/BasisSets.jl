using JSON3
using JSON

function transformbasisname(root, name, version="1")
    name = lowercase(name)
    name = string(name, ".$version.table.json")
    
    basis = joinpath(root, "data", name)

    path = joinpath(root, "data")

    #name = replace(name, "/" => "_sl_")
    #name = replace(name, "*" => "_st_")

    return [path, basis]
end

function _getelement(data, element)
    return data["elements"][element]
end

function getdir(dir, element)
    path = dir[1]
    basis = dir[2]
    res = JSON3.read(basis)
    
    databranchpath = joinpath(path, res["elements"][element])
    databranch = JSON3.read(databranchpath)["elements"][element]["components"]

    data = JSON3.read(joinpath(root, "data", databranch[1]))

    out = _getelement(data, element)

    return out
end

root = pwd()

final = transformbasisname(root, "MIDI", "0")

res = getdir(final, 2)

println("$(root)/data/METADATA.json")

function readjson(file)
    open(file,"r") do f
        return JSON.parse(f)
    end
end

res = readjson("$(root)/data/METADATA.json")
println(res["cc-pv(5+d)z"])

v0 = []
v1 = []

keys0 = []
keys1 = []

function _getbasename(metadata, key, version)
    name = metadata[key]["versions"][version]["file_relpath"]
    basename = split(name, ".")[1]

    return basename
end

for key in keys(res)
    versions = keys(res[key]["versions"])

    for version in versions
        if version == "0"
            push!(keys0, key)
            basename = _getbasename(res, key, version)

            push!(v0, basename)
        else
            push!(keys1, key)
            basename = _getbasename(res, key, version)

            push!(v1, basename)
        end
    end
end

function _getversionfile(keys, v, filename)

    basis = Dict(keys .=> v)
    comparison = []

    file = open(filename, "w")

    for i in eachindex(keys)
        push!(comparison, cmp(lowercase(keys[i]), lowercase(v[i])))

        if cmp(lowercase(keys[i]), lowercase(v[i])) != 0
            println("$(keys[i]) != $(v[i])")
            write(file, "$(lowercase(keys[i])) != $(lowercase(v[i]))\n")
        end
    end

    close(file)
end

_getversionfile(keys0, v0, "diffV0.txt")
_getversionfile(keys1, v1, "diffV1.txt")