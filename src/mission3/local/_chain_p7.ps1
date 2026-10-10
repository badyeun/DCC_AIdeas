Set-Location C:\DevBox\projects\DCC_AIdeas
$log = 'C:\DevBox\projects\DCC_AIdeas\results\mission3\local\_p5_console.log'
while (-not (Select-String -Path $log -Pattern 'PHASE5_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\DevBox\projects\DCC_AIdeas\src\mission3\local\run_phase7.ps1
