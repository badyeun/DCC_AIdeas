# Chain: wait for wave two, then confirm roberta-large.
Set-Location C:/DevBox/projects/DCC_AIdeas
$log = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/_w2.log'
while (-not (Select-String -Path $log -Pattern 'W2_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_large_confirm.ps1
