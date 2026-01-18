# run_sweep.ps1

# 1. Ensure the results directory exists
if (!(Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

$csvPath = "results/performance.csv"

# 2. Initialize the CSV with headers if it doesn't exist
if (!(Test-Path $csvPath)) {
    "version,M,N,K,seconds,gflops" | Out-File -FilePath $csvPath -Encoding ascii
}

# 3. Define the sizes to test
# We start small and move to larger sizes to see the "Cache Cliff"
$sizes = @(32, 64, 128, 256, 512, 1024, 2048)

Write-Host "--- Starting MatMul Performance Sweep (-O3) ---" -ForegroundColor Cyan

foreach ($n in $sizes) {
    Write-Host "Testing N=$n... " -NoNewline -ForegroundColor White
    
    # Execute the binary with M N K arguments
    # .bin\bench-matmul.exe is the target from your Makefile
    & "./bin/bench-matmul.exe" $n $n $n
    
    # The .exe handles the actual timing and CSV logging internally
}

Write-Host "--- Sweep Complete ---" -ForegroundColor Green
Write-Host "Results saved to: $csvPath" -ForegroundColor Gray