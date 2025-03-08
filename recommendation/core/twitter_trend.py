import time
import math

class TwitterStyleTrending:
    def __init__(self):
        # Configuration constants
        self.LONG_TERM_WINDOW = 30 * 60 * 1000  # 30 minutes for baseline
        self.SHORT_TERM_WINDOW = 60 * 1000      # 1 minute for current activity
        self.DECAY_FACTOR = 0.5                 # Half-life decay factor
        self.MIN_OCCURRENCES = 2                # Minimum occurrences to be considered trending
        
        # Store word occurrences with timestamps
        self.words = {}
        
        # Store historical baseline rates
        self.baseline_rates = {}
        
        # Store EMA (Exponential Moving Average) values
        self.ema_values = {}

    def add_word(self, word):
        """
        Add a word occurrence to the system
        
        Args:
            word (str): The word to add
        """
        now = int(time.time() * 1000)  # Current time in milliseconds
        
        # Initialize or get existing word data
        if word not in self.words:
            self.words[word] = []
            self.baseline_rates[word] = 0
            self.ema_values[word] = 0
        
        # Add new occurrence
        occurrences = self.words[word]
        occurrences.append(now)
        
        # Clean up old occurrences
        cutoff = now - self.LONG_TERM_WINDOW
        while occurrences and occurrences[0] < cutoff:
            occurrences.pop(0)
        
        # Update baseline and EMA
        self.update_metrics(word, now)

    def update_metrics(self, word, now):
        """
        Update baseline rate and EMA for a word
        
        Args:
            word (str): The word to update
            now (int): Current timestamp
        """
        occurrences = self.words[word]
        
        # Calculate current rate (occurrences per minute in short term window)
        recent_occurrences = len([t for t in occurrences if t > now - self.SHORT_TERM_WINDOW])
        current_rate = recent_occurrences / (self.SHORT_TERM_WINDOW / 60000)
        
        # Update baseline rate (long-term average)
        total_occurrences = len(occurrences)
        baseline_rate = total_occurrences / (self.LONG_TERM_WINDOW / 60000)
        self.baseline_rates[word] = baseline_rate
        
        # Update EMA
        old_ema = self.ema_values[word]
        new_ema = (current_rate * self.DECAY_FACTOR) + (old_ema * (1 - self.DECAY_FACTOR))
        self.ema_values[word] = new_ema

    def calculate_trending_score(self, word, now):
        """
        Calculate trending score for a word
        
        Args:
            word (str): The word to calculate score for
            now (int): Current timestamp
            
        Returns:
            float: The trending score
        """
        occurrences = self.words[word]
        baseline_rate = self.baseline_rates[word]
        ema_value = self.ema_values[word]
        
        # Recent occurrences count
        recent_count = len([t for t in occurrences if t > now - self.SHORT_TERM_WINDOW])
        
        if recent_count < self.MIN_OCCURRENCES:
            return 0
        
        # Calculate velocity (rate of change)
        velocity = max(0, ema_value - baseline_rate)
        
        # Calculate recency factor
        last_occurrence = max(occurrences)
        recency = math.exp(-(now - last_occurrence) / self.SHORT_TERM_WINDOW)
        
        # Combine factors
        return (velocity * recency * math.log2(1 + recent_count))

    def get_trending_words(self, limit=5):
        """
        Get current trending words
        
        Args:
            limit (int): Maximum number of words to return
            
        Returns:
            list: Array of trending words with scores
        """
        now = int(time.time() * 1000)
        scores = []
        
        for word, occurrences in self.words.items():
            if len(occurrences) >= self.MIN_OCCURRENCES:
                score = self.calculate_trending_score(word, now)
                recent_count = len([t for t in occurrences if t > now - self.SHORT_TERM_WINDOW])
                
                if score > 0:
                    scores.append({
                        'word': word,
                        'score': score,
                        'recent_count': recent_count,
                        'total_count': len(occurrences),
                        'baseline_rate': self.baseline_rates[word],
                        'ema_value': self.ema_values[word],
                        'last_seen': f"{round((now - max(occurrences)) / 1000)}s ago"
                    })
        
        return [{**item, 'score': f"{item['score']:.3f}"} 
                for item in sorted(scores, key=lambda x: x['score'], reverse=True)[:limit]]


# Test implementation
def simulate_traffic():
    trending = TwitterStyleTrending()
    print("Starting simulation...")
    
    # Add some baseline words
    trending.add_word("hello")
    trending.add_word("world")
    
    print("\nInitial state:", trending.get_trending_words())
    # Add a burst of "trending"
    for _ in range(2):
        trending.add_word("hello")
    trending.add_word("banane")
    
    print("\nAfter more trending:", trending.get_trending_words())

    
    # Add more occurrences over time
    time.sleep(2)
    
    for _ in range(2):
        trending.add_word("trending")
    
    print("\nAfter more trending:", trending.get_trending_words())


if __name__ == "__main__":
    simulate_traffic()