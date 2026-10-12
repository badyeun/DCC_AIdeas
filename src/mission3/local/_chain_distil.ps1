# Chain: wait for the teacher probabilities, then distil.
Set-Location C:/DevBox/projects/DCC_AIdeas
$f = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/soft_targets_ens3.npy'
while (-not (Test-Path $f)) { Start-Sleep -Seconds 15 }
Start-Sleep -Seconds 15
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_distil.ps1
