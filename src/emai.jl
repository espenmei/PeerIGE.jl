using Printf
using CSV, DataFrames, LinearAlgebra, SparseArrays, Statistics
using IterativeSolvers
using JLD2
using Random
import Base: *, eltype, size
import LinearAlgebra: mul!, ldiv!

# Forhindre OpenBLAS-tråder fra å krasje med Julia-tråder på Windows VM
LinearAlgebra.BLAS.set_num_threads(1)

# =====================================================================
# PeerIGE.jl — Social Genetic Effects AI-REML
# This file is a thin façade. Implementation is split across:
#   emai_core.jl    — shared types and observed-data / group construction
#   emai_mme.jl     — matrix-free mixed model equations, CG solves, V_i ops
#   emai_trace.jl   — stochastic trace probes (Hutchinson randomisation)
#   emai_aireml.jl  — score/AI assembly and the run_aireml driver
# The using/import statements above are shared by every included file.
# =====================================================================

include(joinpath(@__DIR__, "emai_core.jl"))
include(joinpath(@__DIR__, "emai_mme.jl"))
include(joinpath(@__DIR__, "emai_trace.jl"))
include(joinpath(@__DIR__, "emai_aireml.jl"))
