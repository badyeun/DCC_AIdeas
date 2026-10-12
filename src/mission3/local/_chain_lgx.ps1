# Chain: wait for the three-member large queue, then the other two.
Set-Location C:/DevBox/projects/DCC_AIdeas
$log = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/_lgf.log'
while (-not (Select-String -Path $log -Pattern 'LGF_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_large_extra.ps1
