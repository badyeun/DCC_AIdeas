$base = 'C:\DevBox\projects\DCC_AIdeas'
while (-not (Select-String -Path "$base\results\mission3\local\_phaseb_console.log" -Pattern 'phase B 종료' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
"=== phase B 종료 감지 $(Get-Date -Format 'HH:mm:ss') — phase C 시작 ==="
powershell -NoProfile -ExecutionPolicy Bypass -File "$base\src\mission3\local\run_phase_c.ps1" *> "$base\results\mission3\local\_phasec_console.log"
"=== phase C 종료 $(Get-Date -Format 'HH:mm:ss') ==="
