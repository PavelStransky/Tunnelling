# Installs the packages listed in Project.toml into the environment of this folder and precompiles them.
# Run once:  julia setup.jl
# Then run the scripts with this environment:  julia --project -t auto <script>.jl

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()

println("\nThe environment in ", @__DIR__, " is ready. Run the scripts from this folder as\n",
        "    julia --project -t auto <script>.jl")
