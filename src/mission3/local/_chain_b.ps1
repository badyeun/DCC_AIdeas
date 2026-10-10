$base = 'C:\DevBox\projects\DCC_AIdeas'
$console = "$base\results\mission3\local\_lr_console.log"
while (-not (Select-String -Path $console -Pattern '학습률 탐색 종료' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
"=== 학습률 탐색 종료 감지 $(Get-Date -Format 'HH:mm:ss') — phase B 시작 ==="
powershell -NoProfile -ExecutionPolicy Bypass -File "$base\src\mission3\local\run_phase_b.ps1" *> "$base\results\mission3\local\_phaseb_console.log"
"=== phase B 종료 $(Get-Date -Format 'HH:mm:ss') ==="
