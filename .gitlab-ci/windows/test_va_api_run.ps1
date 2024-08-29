
function Deploy-Dependencies {
  param (
    [string] $deploy_directory
  )
  
  Write-Host "Copying libva runtime and driver at:"
  Get-Date

  # Copy the VA runtime binaries from the mesa built dependencies so the versions match with the built mesa VA driver binary
  $depsInstallPath="C:\mesa-deps"
  Copy-Item "$depsInstallPath\bin\test_va_api.exe" -Destination "$deploy_directory\test_va_api.exe"
  Copy-Item "$depsInstallPath\bin\vainfo.exe" -Destination "$deploy_directory\vainfo.exe"
  Copy-Item "$depsInstallPath\bin\va_win32.dll" -Destination "$deploy_directory\va_win32.dll"
  Copy-Item "$depsInstallPath\bin\va.dll" -Destination "$deploy_directory\va.dll"

  # Copy Agility SDK into D3D12 subfolder of test_va_api
  New-Item -ItemType Directory -Force -Path "$deploy_directory\D3D12" | Out-Null
  Copy-Item "$depsInstallPath\bin\D3D12\D3D12Core.dll" -Destination "$deploy_directory\D3D12\D3D12Core.dll"
  Copy-Item "$depsInstallPath\bin\D3D12\d3d12SDKLayers.dll" -Destination "$deploy_directory\D3D12\d3d12SDKLayers.dll"

  # Copy WARP next to test_va_api
  Copy-Item "$depsInstallPath\bin\d3d10warp.dll" -Destination "$deploy_directory\d3d10warp.dll"

  Write-Host "Copying libva runtime and driver finished at:"
  Get-Date
}

# Set testing environment variables
$successful_run=1
$testing_dir="$PWD\_install\bin" # vaon12_drv_video.dll is placed on this directory by the build
$test_va_api_app_path = "$testing_dir\test_va_api.exe"
$vainfo_app_path = "$testing_dir\vainfo.exe"

# Deploy test_va_api and dependencies
Deploy-Dependencies -deploy_directory $testing_dir

# Set VA runtime environment variables
$env:LIBVA_DRIVER_NAME="vaon12"
$env:LIBVA_DRIVERS_PATH="$testing_dir"

Write-Host "LIBVA_DRIVER_NAME: $env:LIBVA_DRIVER_NAME"
Write-Host "LIBVA_DRIVERS_PATH: $env:LIBVA_DRIVERS_PATH"

# Print VAAPI support info
Invoke-Expression "$vainfo_app_path -a --display win32 --device help"
Invoke-Expression "$vainfo_app_path -a --display win32 --device 0"

# Run tests
$test_va_api_run_cmd = "$test_va_api_app_path"
Write-Host "Running: $test_va_api_run_cmd"
$test_va_api_ret_code= Invoke-Expression $test_va_api_run_cmd
if (-not($test_va_api_ret_code)) {
  Exit 1
}
