class Mission:
    def __init__(self, id: int, title: str, description: str, objective_type: str, learning_objective: str, chapter_link: str, max_moves: int = None, target_drop: float = None, target_increase: float = None):
        self.id = id
        self.title = title
        self.description = description
        self.objective_type = objective_type
        self.learning_objective = learning_objective
        self.chapter_link = chapter_link
        self.max_moves = max_moves
        self.target_drop = target_drop
        self.target_increase = target_increase

def get_missions():
    """Returns the list of available educational missions."""
    return [
        Mission(1, "First Contact", "Mask a background region using up to 3 moves to see how the model reacts.", "FIRST_MOVE", 
                "Observe that altering the input tensor changes the output prediction.", "Chapter 6: Masking & Predictions", max_moves=3),
                
        Mission(2, "Find the Critical Feature", "Reduce the original predicted score by at least 15% using up to 3 moves.", "REDUCE_SCORE", 
                "Understand that the CNN relies heavily on specific spatial patterns (like faces).", "Chapter 7: The AI X-Ray", max_moves=3, target_drop=15.0),
                
        Mission(3, "The Artifact Test", "Drop the confidence score by at least 20% using up to 3 moves.", "REDUCE_SCORE", 
                "Experience how artificial straight edges confuse the model's filters.", "Chapter 9: Limitations & Bias", max_moves=3, target_drop=20.0),
                
        Mission(4, "Noise Reduction", "Find a region that INCREASES the model's confidence by at least 1% using up to 3 moves.", "INCREASE_SCORE", 
                "Observe that removing distracting background noise (spurious correlations) helps the model focus.", "Chapter 6: Masking & Predictions", max_moves=3, target_increase=1.0)
    ]

def check_mission_complete(mission, original_score, current_score, move_count):
    if move_count == 0:
        return False, "Awaiting your first move."
        
    if mission.max_moves and move_count > mission.max_moves:
        return False, f"Move limit exceeded! You used {move_count}/{mission.max_moves} moves. Click 'Reset Image' to try again."

    if mission.objective_type == "FIRST_MOVE":
        if move_count >= 1:
            return True, "Great! You made your first move and observed the change."
            
    elif mission.objective_type == "REDUCE_SCORE":
        drop = original_score - current_score
        if drop >= mission.target_drop:
            return True, f"Success! You reduced the score by {drop:.1f}%, exceeding the {mission.target_drop}% goal."
            
    elif mission.objective_type == "INCREASE_SCORE":
        increase = current_score - original_score
        if increase >= mission.target_increase:
            return True, f"Success! The score INCREASED by {increase:.1f}%. You successfully removed distracting noise!"
            
    if mission.objective_type == "INCREASE_SCORE":
        return False, f"Keep trying! Current change: {(current_score - original_score):.1f}%"
    else:
        return False, f"Keep trying! Current drop: {(original_score - current_score):.1f}%"
