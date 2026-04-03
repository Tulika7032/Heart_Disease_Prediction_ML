from src.preprocess import load_data, preprocess_data, split_data
from src.train import train_model, save_model
from src.evaluate import evaluate_model

# Load data
df = load_data("data/heart-disease.csv")
print("Data loaded successfully. Shape:", df.shape)

# Preprocess
X, y = preprocess_data(df)
print("Data preprocessed successfully. Shape:", X.shape, y.shape)

# Split
X_train, X_test, y_train, y_test = split_data(X, y)
print("Data split successfully. Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Train & evaluate all models
models = train_model(X_train, y_train)
results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")
    metrics = evaluate_model(model, X_test, y_test)
    results[name] = metrics

    print(f"{name} metrics:", {
    k: v for k, v in metrics.items()
    if k not in ["confusion_matrix", "classification_report"]
})

# Select best model
best_model_name = max(results, key=lambda x: results[x]["f1"])
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name}")

# Save best model
save_model(best_model)
print("Model saved successfully.")