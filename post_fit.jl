# Compare the same data and model under different dilution assumptions.
# Open in VS Code with a Julia REPL started using `julia --project=.`.
# Run Setup and Load first, then run the sections you want, one cell at a time.
# Saving and the expensive likelihood/score calculations are separate sections.

## Setup
if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "src", "emai.jl"))
end
include(joinpath(@__DIR__, "src", "emai_post_compare.jl"))
import Plots

# Change this to the directory containing yX_obs.csv, roster.csv and fits/.
data_dir = "/ess/p805/data/durable/projects/40683815_Espen_IGEpeer/data/edu10_MAT"
runs = [
    "Mean" => "ds_meandilution_class_household",
    "Sum" => "ds_sum_dilution_class_household",
    "Estimated" => "ds_optdilution_class_household",
]

## Load and check the fits
# Full models are loaded one at a time; the retained results are snapshots.
report = compare_fits(data_dir, runs)
figures = Dict{String,Plots.Plot}()
display(select(report.summary, :fit, :psi, :psi_estimated, :psi_se,
    :converged, :iterations, :step_crit))

## Full fit diagnostics
display(report.summary)

## Inspect one fit
# Change to 2 for Sum or 3 for Estimated.
fit_number = 1
display(report.fits[fit_number].result)

## Convergence and psi
figures["convergence"] = comparison_plot(report, "convergence")
display(figures["convergence"])

## Raw parameters
# Social components here use different scales and are not directly comparable.
display(report.raw_parameters)

## Common-scale variance components
# Social components are per peer at the reference group size.
display(report.common_parameters)
figures["components_reference"] = comparison_plot(report, "components_reference")
display(figures["components_reference"])

## Derived quantities
display(report.derived)
figures["derived_reference"] = comparison_plot(report, "derived_reference")
display(figures["derived_reference"])

## Fixed effects
display(report.fixed_effects)
figures["fixed_effects"] = comparison_plot(report, "fixed_effects")
display(figures["fixed_effects"])

## Group-size curves
display(report.curves)
figures["group_size_curves"] = comparison_plot(report, "group_size_curves")
display(figures["group_size_curves"])

## BLUP comparisons
# Statistics use all observed individuals; scatterplots use a reproducible sample.
display(report.blup_comparison)
figures["blup_comparison"] = comparison_plot(report, "blup_comparison")
display(figures["blup_comparison"])

## Sampling correlations
figures["estimate_correlations"] = comparison_plot(report, "estimate_correlations")
display(figures["estimate_correlations"])

## Save tables and the figures created so far
save_comparison(report, figures)

## Optional: stochastic likelihood comparison
# EXPENSIVE: requires full, converged models; run separately in a suitably sized job.
# likelihood = compare_likelihood(report; K=64)
# display(likelihood)

## Optional: psi score tests
# EXPENSIVE: fixed-psi fits only; requires full, converged models.
# scores = compare_psi_scores(report; K=64)
# display(scores)
