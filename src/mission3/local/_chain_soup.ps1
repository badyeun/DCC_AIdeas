# Chain: wait for the Validation probability cache to finish, then train.
Set-Location C:/DevBox/projects/DCC_AIdeas
$dir = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/val_probs'
while ((Get-ChildItem "$dir/*.npy" -ErrorAction SilentlyContinue).Count -lt 25) {
    Start-Sleep -Seconds 20
}
Start-Sleep -Seconds 20
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_soup_train.ps1
