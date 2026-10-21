$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskDocx = Join-Path $taskRoot 'presentation_script_pre_v3_restructured.docx'
$taskPdf = Join-Path $taskRoot 'docs\script_revision_evidence\presentation_script_pre_v3_preview.pdf'
$wordApp = New-Object -ComObject Word.Application
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
try {
    $taskDocument = $wordApp.Documents.Open($taskDocx, $false, $true)
    $taskDocument.Repaginate()
    $taskPages = $taskDocument.ComputeStatistics(2)
    $taskDocument.ExportAsFixedFormat($taskPdf, 17)
    $taskDocument.Close(0)
    Write-Output "Word opened DOCX successfully; pages: $taskPages; PDF: $taskPdf"
} finally {
    $wordApp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null
}
