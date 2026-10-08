@echo off

echo =============================================
echo Starting Sequential Training Pipeline (5-fold CV)
echo =============================================

for %%M in (cnn vit cnn_svm hybrid) do (
    echo.
    echo [RUNNING] Starting training for model: %%M...
    
    :: Just run python normally. The Python script will handle saving the logs.
    python modeling/oasis-cnn/src/train.py --model %%M --cv || (
        echo [ERROR] Training failed for model: %%M
        exit /b 1
    )
    
    echo [SUCCESS] Finished training for model: %%M
)

echo.
echo =============================================
echo All models trained successfully!
echo =============================================
