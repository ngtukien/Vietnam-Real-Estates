# Xuất từng slide của một pptx ra PNG bằng PowerPoint (COM, không mở cửa sổ) để duyệt bố cục.
# Dùng: & slide\build\render.ps1 -Pptx <file.pptx> -OutDir <thư mục> [-Width 1600] [-Slides 3,5]
param([Parameter(Mandatory)][string]$Pptx, [Parameter(Mandatory)][string]$OutDir, [int]$Width = 1600, [int[]]$Slides)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open((Resolve-Path $Pptx).Path, $true, $false, $false)
    $height = [int]($Width * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
    foreach ($s in $pres.Slides) {
        if ($Slides -and ($Slides -notcontains $s.SlideIndex)) { continue }
        $s.Export((Join-Path (Resolve-Path $OutDir).Path ('s{0:D2}.png' -f $s.SlideIndex)), 'PNG', $Width, $height)
    }
    $n = $pres.Slides.Count
    $pres.Close()
    Write-Host "rendered $n slides to $OutDir"
} finally {
    $app.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
