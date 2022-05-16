param(
    [parameter(Mandatory=$true)][ValidateRange(1, 1000)]
    [int]$repeats,

    [parameter(Mandatory=$false)]
    [string]$logFilename
)

If ($repeats -eq 0)
{
    $repeats = 1
}

If (!(Test-Path -Path "train"))
{
    New-Item -Path "train" -ItemType Directory
}

If ($logFilename.Length -eq 0)
{
    Invoke-Expression -Command "python .\main.py train data -v --log train\train.csv"
    For ($i = 2; $i -le $repeats; $i++)
    {
        Invoke-Expression -Command "python .\main.py train data --input model.pt -v --log train\train.csv --append-log"
    }
}
Else
{
    Invoke-Expression -Command "python .\main.py train data -v --log train\train.csv > train\$logFilename$i.txt"
    For ($i = 2; $i -le $repeats; $i++)
    {
        Invoke-Expression -Command "python .\main.py train data --input model.pt -v --log train\train.csv --append-log > train\$logFilename$i.txt"
    }
}

