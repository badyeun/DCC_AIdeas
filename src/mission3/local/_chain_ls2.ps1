# Wait for part one, then start part two.
#
# Counting history files was wrong the first time: all four teacher folds share
# one tag and overwrite a single history_..._T.json, so that count stops at 1.
# The fold predictions carry the fold number, and the seven part-one
# experiments each have their own tag, so count those instead.
Set-Location C:\DevBox\projects\DCC_AIdeas
$dir = 'C:\DevBox\projects\DCC_AIdeas\results\mission3\local'
while ((Get-ChildItem "$dir\history_*_LS.json" -ErrorAction SilentlyContinue).Count -lt 7) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\DevBox\projects\DCC_AIdeas\src\mission3\local\run_labelside2.ps1
