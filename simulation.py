import random
import math

def monte_carlo_simulation(num_simulations):
    # Define the deck with all suits and each card appearing twice
    suits = ['S', 'C', 'H', 'D']
    ranks = ['10', 'J', 'Q', 'K', 'A']
    deck = [rank + suit for rank in ranks for suit in suits] * 2
    target_cards = ['KH', 'KH', 'AH', 'AH']
    success_count = 0

    for _ in range(num_simulations):
        # Shuffle the deck
        random.shuffle(deck)

        # Deal the cards to 4 players
        players = [deck[i*10:(i+1)*10] for i in range(4)]

        # Check if each player has exactly one of the target cards
        target_counts = [sum(1 for card in player if card in target_cards) for player in players]
        if target_counts == [1, 1, 1, 1]:
            success_count += 1

    probability = success_count / num_simulations
    standard_error = math.sqrt(probability * (1 - probability) / num_simulations)
    return probability, standard_error

# Run the simulation with a large number of trials
num_simulations = 100000
probability, standard_error = monte_carlo_simulation(num_simulations)
print(f"Estimated probability: {probability:.6f}")
print(f"Standard error: {standard_error:.6f}")
print(f"95% confidence interval: [{probability - 1.96 * standard_error:.6f}, {probability + 1.96 * standard_error:.6f}]")
