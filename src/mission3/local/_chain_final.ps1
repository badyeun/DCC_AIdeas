# Chain: re-ship the three large members first, then resume exploration.
Set-Location C:/DevBox/projects/DCC_AIdeas
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_large_reship.ps1
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_large_wave2.ps1
