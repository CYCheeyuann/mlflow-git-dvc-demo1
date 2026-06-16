import yaml
import joblib
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_params(path="params.yaml"):
    with open(path, "r") as file:
        params = yaml.safe_load(file)
    return params

def main():
    # 1. Load parameters to get the identical random seed
    params = load_params()
    model_params = params["model"]

    # 2. Load dataset
    df = pd.read_csv("data/iris.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    # 3. Re-create the exact same test split using the identical seed
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=model_params["random_state"],
        stratify=y
    )

    # 4. Load the trained model from the local models/ folder
    model_path = "models/model.pkl"
    model = joblib.load(model_path)

    # -----------------------------------------------------------------
    # OPTION B: Load from mlruns/ folder instead (Commented out)
    # -----------------------------------------------------------------
    # run_id = "YOUR_ACTUAL_RUN_ID_HERE" 
    # model_uri = f"runs:/{run_id}/model-rf"
    # model = mlflow.sklearn.load_model(model_uri)
    # -----------------------------------------------------------------

    # 5. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 6. Calculate and print the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Inference Test Accuracy: {accuracy:.4f}\n")

    # 7. Compare predictions with ground truth for a few sample cases
    print("--- Sample Predictions vs Ground Truth ---")
    num_samples = 5
    
    # Reset index of y_test so it aligns perfectly with the NumPy array y_pred
    y_test_reset = y_test.reset_index(drop=True)
    
    for i in range(num_samples):
        print(f"Sample {i+1}: Predicted Label = {y_pred[i]}, Ground Truth = {y_test_reset[i]}")

if __name__ == "__main__":
    main()