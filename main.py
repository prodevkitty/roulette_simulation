from modules.data_loader import load_and_preprocess_data, load_player_data
from modules.model_trainer import train_model, retrain_model

def main():
    try: 
        # Load and preprocess data
        player_data, bank_data = load_player_data('data')
        preprocessed_data = load_and_preprocess_data('data/player_data.csv')
        train_model(preprocessed_data)        

    except KeyboardInterrupt:
        print("Script interrupted by user. Exiting gracefully...")
        # Perform any necessary cleanup here
        # For example, save the current state, close files, etc.
        exit(0)

if __name__ == "__main__":
    main()