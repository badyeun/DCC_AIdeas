$b = 'C:\DevBox\projects\DCC_AIdeas'
while (-not (Select-String -Path "$b\results\mission3\local\_phased_console.log" -Pattern 'PHASE_D_DONE' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 30 }
'[CHAIN] phase D done, starting FINAL'
powershell -NoProfile -ExecutionPolicy Bypass -File "$b\src\mission3\local\run_final.ps1" *> "$b\results\mission3\local\_final_console.log"
'[CHAIN] FINAL done'
