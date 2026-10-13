# Chain: once the split-teacher students finish, compute their Validation
# probabilities so the comparison can be made without another GPU pass.
Set-Location C:/DevBox/projects/DCC_AIdeas
$log = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/_d7.log'
while (-not (Select-String -Path $log -Pattern 'D7_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 30
}
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_OFFLINE = '1'
"===== split student Validation probs start $(Get-Date -Format 'HH:mm:ss') ====="
& $PY src/mission3/local/val_probs.py 2>&1 | Where-Object { $_ -notmatch 'it/s\]|unauthenticated' }
"[D8_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
