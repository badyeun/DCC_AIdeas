# Chain: poll a console log for a sentinel, then start the next phase.
#
# Written with forward slashes on purpose. The first two attempts at this
# generator put a backslash immediately before a shell variable, bash ate the
# expansion, and the chains died on their first poll with a literal ${log} in
# the path -- unnoticed because the generator printed its own variables
# instead of the file. PowerShell accepts forward slashes, so the hazard is
# simply removed, and the file is read back before launch.
Set-Location C:/DevBox/projects/DCC_AIdeas
$log = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/_p6_console.log'
while (-not (Select-String -Path $log -Pattern 'PHASE6_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_phase3.ps1
