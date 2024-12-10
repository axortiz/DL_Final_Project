import lightgbm as lgb
import numpy as np
from data_src.data_loader import load_and_process_data

def predict_boxing_matches(model_path, data_path):
    # Load and process the inference data
    X, _, boxer_names = load_and_process_data(data_path, preserve_names=True)
    
    # Load the trained model
    model = lgb.Booster(model_file=model_path)
    
    # Make predictions
    predictions_prob = model.predict(X)
    predictions = np.argmax(predictions_prob, axis=1)
    
    # Convert numeric predictions to human-readable results
    results = []
    for i, (pred, probs) in enumerate(zip(predictions, predictions_prob)):
        first_boxer, second_boxer = boxer_names[i]
        
        if pred == 0:
            winner = first_boxer
            result = "win"
        elif pred == 1:
            winner = second_boxer
            result = "win"
        else:
            winner = "Draw"
            result = "draw"
            
        confidence = float(max(probs)) * 100
        
        match_result = {
            'first_boxer': first_boxer,
            'second_boxer': second_boxer,
            'predicted_winner': winner,
            'result': result,
            'confidence': f"{confidence:.2f}%",
            'win_probabilities': {
                first_boxer: f"{float(probs[0]) * 100:.2f}%",
                second_boxer: f"{float(probs[1]) * 100:.2f}%",
                'Draw': f"{float(probs[2]) * 100:.2f}%"
            }
        }
        results.append(match_result)
    
    return results

def main():
    MODEL_PATH = 'best_model_lgb.txt'
    INFERENCE_DATA_PATH = 'data_src/inference_data.csv'
    
    try:
        results = predict_boxing_matches(MODEL_PATH, INFERENCE_DATA_PATH)
        
        # Print predictions in a formatted way
        print("\n=== Boxing Match Predictions ===\n")
        for i, result in enumerate(results, 1):
            print(f"Match {i}:")
            print(f"🥊 {result['first_boxer']} vs {result['second_boxer']}")
            print(f"Predicted Winner: {result['predicted_winner']} ({result['confidence']} confidence)")
            print("\nWin Probabilities:")
            for boxer, prob in result['win_probabilities'].items():
                print(f"  {boxer}: {prob}")
            print("\n" + "="*40 + "\n")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
