# Chain: wait for ALL_DONE, then run run_phase6.ps1.
# Reordered 14:55 -- phase 6 (distillation with the corrected pos_weight) is
# the arm most likely to pay, so it no longer waits behind three other phases.
Set-Location C:\DevBox\projects\DCC_AIdeas
$log = 'C:\DevBox\projects\DCC_AIdeas\results\mission3\local${log}'
while (-not (Select-String -Path $log -Pattern 'ALL_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\DevBox\projects\DCC_AIdeas\src\mission3\local${script}
