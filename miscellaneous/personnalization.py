import heapq
import numpy as np
from datetime import datetime

class UserSessionRecommender(object):
    """
    A class that defines a user recommendation that promotes 
        recently and frequently explored categories that were
        explored historically or throughout a session.
    ---
    @author: Phileas Hocquard
    """
    user_id = None
    session_id = None
    ema_scores = {}
    ema_settings = {}

    def __init__(self, session_id=None, user_id=None, event=None, ema_settings={}, **kargs) -> None:
        """
        user_id: None or integer,  if given, our user has already an ema or has history
        session_id: None or integer, depending on how we store the ema on the session could retrieve it similarly to user_id
        event: dictionnary,  {'category': 'Category Name', 'value': interaction score, 'timestamp':creation timestamp}
        ema_settings: dictionnary, parameters for how we update the moving average.

        Initialization function to create a usersession recommendation object
        """
        if user_id:
            history_or_ema, object_type, ema_settings = self.get_history_or_ema_info(user_id)
            if object_type:
                self.ema_scores = self.calculate_ema_history(history_or_ema, **ema_settings)
                self.ema_settings = ema_settings
            else:
                self.ema_scores = history_or_ema
        if event:
            self.ema_settings = ema_settings or self.ema_settings
            self.ema_category_update(**(event + self.ema_settings))
        self.session_id = self.session_id or session_id


    def ema_category_update(self, category=None, value=1, alpha=0.1, penalty_round=False, mode=None, timestamp=None, history_length=None, **kwargs):
            """
            Computes the Exponential Moving Average for a single category.
            category: string, the item's category, tokenization or topic class.
            value: float, the value of the category, as a single update this will serve a means of translating an action into a score
            alpha: float, the rate of the update, this is where the exponential relation lives.
			penalty_round: boolean, penalizing previously observed categories
            mode: string, serves as an indication of the weight put in place. 
            We either use no weight, the date at which the event's category was called, or the length of the history for regularizing.

            Updates the ema map for our category scores
            """

            def time_decay(timestamp, rate=0.1, scale=(12 * 3600)):
                now = datetime.now()
                delta_t = (now - datetime.fromtimestamp(timestamp)).total_seconds()
                return np.exp(- rate * delta_t / scale)

            if mode == 'last_explored':
                weight = time_decay(timestamp)
            elif mode == 'length':
                weight =  1 - (1 / float(history_length))
            else:
                weight = 1

            if category:
                if category not in self.ema_scores:
                    self.ema_scores[category] = {}
                    self.ema_scores[category]['score'] = value
                else:
                    self.ema_scores[category]['score'] = \
                        alpha * value * weight + (1 - alpha) * self.ema_scores[category]['score']

                if timestamp:
                    self.ema_scores[category]['timestamp'] = timestamp
                elif 'timestamp' not in self.ema_scores[category]:
                        self.ema_scores[category]['timestamp'] = None

            if penalty_round:
                # Penalizing all items (except for the current category if mentioned)
                 #Note: A faster update could be done if we stored the items under a sparse vector and compute an inner dot product
				# between a binary vector and the observed and unobserved categories. : user item sparse <123:0.9, 9:.3,..> += vector binary sparse vector <9:1> * alpha * decay * user item vector <123:0.9 ..>  
				# negative_fill for historical data is not optimal O(n*m), should use a grouping strategy.
                for item_category, values in list(self.ema_scores.items()):
                    if item_category != category:
                        self.ema_category_update(category=item_category, value=0.5, alpha=alpha, mode=mode, history_length=len(self.ema_scores), **values)


    def calculate_ema_history(self, history, **kwargs):
        """
        Calculates the Exponential Moving Average (EMA) of a user's history.
        
        history: list of dictionaries, where each dictionary represents a product or category
                the user has interacted with, and has the following format:
                {'category': 'Category Name', 'value': interaction score, 'timestamp':creation timestamp}
        **kwargs: dictionary of parameters with their respective values. 
        
        Updates the users EMA categories
        """
        self.ema_scores = {}
        now = datetime.now()
        for interaction in history:

            category = f"{interaction['vertical']}-{interaction['category']}"
            value = interaction['value']
            timestamp = interaction['creation_date']
            self.ema_category_update(category=category, value=value, timestamp=timestamp, history_length=len(history), **kwargs)


    def set_ema_to_top_n(self, n=50):
        """
        Resets the ema score to the top n categories with the highest EMA scores.
        n: int, the number of categories to return
    
        """
        self.ema_scores = heapq.nlargest(n, self.ema_scores)

    
    def get_history_or_ema_info(self, user_id)->(dict, bool, dict): # an SQL or NoSQL query retrieving previous user ema or history
        pass

history = [
    {'vertical': 'education', 'category': 'business_growth', 'value': 1, 'creation_date': 1675826939}, # 1 month ago..
    {'vertical': 'business&money', 'category': 'personal_finance', 'value': 1, 'creation_date': 1673148539},
    {'vertical': 'business&money', 'category':'investing', 'value': 1, 'creation_date': 1675826929},
    {'vertical': 'business&money', 'category':'investing', 'value': 2, 'creation_date': 1675826939},
    {'vertical': 'business&money', 'category': 'personal_finance', 'value': 2, 'creation_date': 1675913339}
]

parameters =  {'penalty_round': True, 'alpha': 0.1, 'mode': 'last_explored'}
user_session_recommender = UserSessionRecommender()
user_session_recommender.calculate_ema_history(history=history, **parameters)
# print(user_session_recommender.ema_scores)
# parameters =  {'penalty_round': True, 'alpha': 0.1, 'mode': 'last_explored'}
user_session_recommender.ema_category_update(penalty_round=True)
print(user_session_recommender.ema_scores)

#We have user sessions
#We have session recommendations



# limit to last 50 categories (category-vertical) CV
# limit to last 100 topic words

# (sessionID, userID, CV), last_visited, frequency, ema


# no view = reduce
# view = add

# item_value = 1
# prev_ema = 0
# (item_value) * alpha + (1-alpha) * prev_ema 
# 0.7

# item_value = 1
# prev_ema = 0.7
# (item_value) * alpha + (1-alpha) * prev_ema 
# 0.91


# item_value = 0
# prev_ema = 0.91
# (item_value) * alpha + (1-alpha) * prev_ema 
# 0.91

# # (0, 1, None) = (current_val) - (current_val/l)

# # x = .99



# 0.7 + .3 * p | i + a * p
# = i + a * (i + a * p)

# i + ai + a^2*p
# = i + a * (i + a * (i + a * p))

# f3(a,i,p) = i + ai + a^2*i + a^3*p
# f(x) = i + a * f-1(x)

# = 0.7 + .3 * (0.7 + .3 * (0.7 + .3 * (0.7 + .3 * p)))
# i + ai + a^


# item_value = 0 # not viewed
# (0 + item_value) * alpha + (1-alpha) * prev_ema 
# 0 + 0.3 * 0.91

#  user_embedding = np.zeros(item_embeddings.shape[1])
#     alpha = 0.5
#     for item, time in user_history:
#         elapsed_time = current_time - time
#         item_embedding = item_embeddings[item]
#         user_embedding = alpha * user_embedding + (1 - alpha) * item_embedding / (1 + elapsed_time)