param(
    [string]$Experiment = "configs/experiments/runtime-classical.yaml",
    [string]$LocalConfig = "",
    [string]$Python = "python"
)

$args = @("-m", "real_time_ml", "--experiment", $Experiment)
if ($LocalConfig) {
    $args += @("--local-config", $LocalConfig)
}
$args += "report"

# This creates only the run manifest and the normalized final report.  It does
# not read raw data, train a model, or touch legacy artifacts.
& $Python @args
exit $LASTEXITCODE
