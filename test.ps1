$imgFolder = Resolve-Path $args[0]
$testImages = Get-ChildItem -Path $imgFolder -Name
foreach ($image in $testImages)
{
    Invoke-Expression "pythonw main.py classify $imgFolder/$image --show"
}