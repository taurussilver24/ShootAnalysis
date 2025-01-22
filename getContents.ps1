# Function to recursively build the directory tree
function Get-DirectoryTree {
    param (
        [string]$Path,
        [int]$Indent = 0
    )

    # Get all items in the current directory
    $items = Get-ChildItem -Path $Path

    foreach ($item in $items) {
        # Print the current item with indentation
        Write-Host (" " * ($Indent * 4)) "├── $($item.Name)"

        # If the item is a directory, recurse into it
        if ($item.PSIsContainer) {
            Get-DirectoryTree -Path $item.FullName -Indent ($Indent + 1)
        }
    }
}

# Get the current directory
$rootDir = Get-Location

# Print the root directory
Write-Host $rootDir

# Generate the directory tree
Get-DirectoryTree -Path $rootDir

# Pause to prevent the window from closing
Write-Host "Press any key to exit..."
[System.Console]::ReadKey() | Out-Null
