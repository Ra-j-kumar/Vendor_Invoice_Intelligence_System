from pathlib import Path
import joblib

from data_prepossing import load_vendor_invoice_data , preapare_feature , split_data
from model_evalution import (
    train_linear_model,
    train_decision_tree,
    train_random_forest,
    evalute_model
)

def main():
    db_path = "../data/inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    #Load Data
    df = load_vendor_invoice_data(db_path)
    
    #Prepare Data
    X , y = preapare_feature(df)
    X_train , X_test , y_train , y_test = split_data(X,y)
    
    # Train Models
    lr_model = train_linear_model(X_train , y_train)
    dt_model = train_decision_tree(X_train , y_train)
    rf_model = train_random_forest(X_train , y_train)
    
    # Evalute Models
    results = []
    results.append(evalute_model(lr_model , X_test , y_test , "Linear Regression"))
    results.append(evalute_model(dt_model , X_test , y_test , "Decision Tree Regression"))
    results.append(evalute_model(rf_model , X_test , y_test , "Random Forest Regression"))
    
    print(results)
    # Select best model (lowest MAE)
    best_model_info = min(results , key=lambda x: x['mae'])    
    best_model_name = best_model_info["model_name"]    
    
    best_model = {
        "Linear Regression" : lr_model,
        "Decision Tree Regression" : dt_model,
        "Random Forest Regression" : rf_model
    }[best_model_name]
    
    # save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model , model_path)
    
    print(f"\nBest model saved : {best_model_name}")
    print(f"Model Path: {model_path}")
    
if __name__ == "__main__":
    main()
    