import nltk

from model.models import UserModelSession, Choice, UserModelRun, Protocol
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime

nltk.download("wordnet")
from nltk.corpus import wordnet  # noqa


class ModelDecisionMaker:
    def __init__(self):

        self.data = pd.read_csv('/Users/lisaxy/final_project/model/empatheticPersonas.csv') #change path

        # Titles from workshops (Title 7 adapted to give more information)
        self.PROTOCOL_TITLES = [
            "0: None",
            "1: Connecting with the Child [Week 1]",
            "2: Laughing at our Two Childhood Pictures [Week 1]",
            "3: Falling in Love with the Child [Week 2]",
            "4: Vow to Adopt the Child as Your Own Child [Week 2]",
            "5: Maintaining a Loving Relationship with the Child [Week 3]",
            "6: An exercise to Process the Painful Childhood Events [Week 3]",
            "7: Protocols for Creating Zest for Life [Week 4]",
            "8: Loosening Facial and Body Muscles [Week 4]",
            "9: Protocols for Attachment and Love of Nature  [Week 4]",
            "10: Laughing at, and with One's Self [Week 5]",
            "11: Processing Current Negative Emotions [Week 5]",
            "12: Continuous Laughter [Week 6]",
            "13: Changing Our Perspective for Getting Over Negative Emotions [Week 6]",  # noqa
            "14: Protocols for Socializing the Child [Week 6]",
            "15: Recognising and Controlling Narcissism and the Internal Persecutor [Week 7]",  # noqa
            "16: Creating an Optimal Inner Model [Week 7]",
            "17: Solving Personal Crises [Week 7]",
            "18: Laughing at the Harmless Contradiction of Deep-Rooted Beliefs/Laughing at Trauma [Week 8]",  # noqa
            "19: Changing Ideological Frameworks for Creativity [Week 8]",
            "20: Affirmations [Week 8]",
        ]

        self.TITLE_TO_PROTOCOL = {
            self.PROTOCOL_TITLES[i]: i for i in range(len(self.PROTOCOL_TITLES))
        }

        self.recent_protocols = deque(maxlen=3)
        self.reordered_protocol_questions = {}
        self.protocols_to_suggest = []

        # Goes from user id to actual value
        self.current_run_ids = {}
        self.current_protocol_ids = {}

        self.positive_protocols = [i for i in range(1, 21)]

        self.INTERNAL_PERSECUTOR_PROTOCOLS = [
            self.PROTOCOL_TITLES[15],
            self.PROTOCOL_TITLES[16],
            self.PROTOCOL_TITLES[8],
            self.PROTOCOL_TITLES[19],
        ]

        # Keys: user ids, values: dictionaries describing each choice (in list)
        # and current choice
        self.user_choices = {}

        # Keys: user ids, values: scores for each question
        #self.user_scores = {}

        # Keys: user ids, values: current suggested protocols
        self.suggestions = {}

        # Tracks current emotion of each user after they classify it
        self.user_emotions = {}

        self.guess_emotion_predictions = {}
        # Structure of dictionary: {question: {
        #                           model_prompt: str or list[str],
        #                           choices: {maps user response to next protocol},
        #                           protocols: {maps user response to protocols to suggest},
        #                           }, ...
        #                           }
        # This could be adapted to be part of a JSON file (would need to address
        # mapping callable functions over for parsing).

        self.QUESTIONS = {
            "opening_prompt": {
                "model_prompt": [["Hello. "],
                ["How are you feeling today?"]],

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(
                        user_id, app, db_session
                    )
                },
                "protocols": {"open_text": []},
            },
            "guess_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_guess_emotion(
                    user_id
                ),
                "choices": {
                    "yes": {
                        "sad": "after_classification_sad",
                        "angry": "after_classification_angry",
                        "scared": "after_classification_scared",
                        "happy": "after_classification_happy",
                    },
                    "no": "check_emotion",
                },
                "protocols": {"yes": [], "no": []},
            },


            "check_emotion": {
                "model_prompt": [ np.random.choice(self.data["All emotions - I am sorry. Please select from the emotions below the one that best reflects what you are feeling:"].dropna().tolist()) ],
                "choices": {
                    "sad": "after_classification_sad",
                    "angry": "after_classification_angry",
                    "scared": "after_classification_scared",
                    "happy": "after_classification_happy",
                },
                "protocols": {"sad": [self.PROTOCOL_TITLES[k] for k in [3, 7, 10, 12, 18]],
                              "angry": [self.PROTOCOL_TITLES[k] for k in [3, 7, 10, 12, 18]],
                              "scared" : [self.PROTOCOL_TITLES[k] for k in [3, 7, 10, 12, 18]],
                              "happy": []},
            },

            #--------------------SADNESS--------------------#

            "after_classification_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Was this caused by a specific event/s?"].dropna().tolist()) ],
                "choices": {
                    "yes": "event_is_recent_sad",
                    "no": "more_questions_sad", #change this. suggest protocols if no event or ask other questions?
                },
                "protocols": {"yes": [], "no": []},
            },

            "event_is_recent_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Was this caused by a recent or distant event (or events)?"].dropna().tolist()) ],
                "choices": {
                    "It was recent": "revisiting_recent_events_sad",
                    "It was distant": "revisiting_distant_events_sad",
                },
                "protocols": {"It was recent": [], "It was distant": []},
            },

            "revisiting_recent_events_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?"].dropna().tolist()) ],
                #+ "{} and found this reignited emotions of old events? ".format(
                #    self.PROTOCOL_TITLES[11]
                #),
                "choices": {
                    "yes": "more_questions_sad",
                    "no": "more_questions_sad",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[8]],
                    "no": [self.PROTOCOL_TITLES[11]],
                },
            },

            "revisiting_distant_events_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?"].dropna().tolist()) ],
                #+ "{} and found this reignited emotions of ".format(
                #    self.PROTOCOL_TITLES[6]
                #)
                #+ "old events?",
                "choices": {
                    "yes": "more_questions_sad",
                    "no": "more_questions_sad",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [self.PROTOCOL_TITLES[6]],
                },
            },

            "more_questions_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Thank you. Now I will ask some questions to understand your situation."].dropna().tolist()) ],
                "choices": {
                    "Okay": lambda user_id, db_session, curr_session, app: self.get_next_question_sad(user_id, app), #"displaying_antisocial_behaviour_sad",
                    "I'd rather not": "suggestions",
                },
                "protocols": {
                    "Okay": [],
                    "I'd rather not": [self.PROTOCOL_TITLES[k] for k in [3, 7, 10, 12, 18]],
                },
            },

            "displaying_antisocial_behaviour_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Have you strongly felt or expressed any of the following emotions towards someone:"].dropna().tolist()),
                "Envy, jealousy, greed, hatred, mistrust, "
                "malevolence, or revengefulness?"],
                "choices": {
                    "yes": "suggestions",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question_sad(user_id, app), #"internal_persecutor_saviour_sad",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]],
                    "no": [],
                },
            },

            "internal_persecutor_saviour_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Do you believe that you should be the saviour of someone else?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "internal_persecutor_victim_sad",
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_victim_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Do you see yourself as the victim, blaming someone else for how negative you feel?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "internal_persecutor_controlling_sad",
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_controlling_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Do you feel that you are trying to control someone?"].dropna().tolist()) ],
                "choices": {
                "yes": "suggestions",
                "no": "internal_persecutor_accusing_sad"
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_accusing_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Are you always blaming and accusing yourself for when something goes wrong?"].dropna().tolist()) ],
                "choices": {
                "yes": "suggestions",
                "no": lambda user_id, db_session, curr_session, app: self.get_next_question_sad(user_id, app), #"rigid_thought_sad"
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "rigid_thought_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - In previous conversations, have you considered other viewpoints presented?"].dropna().tolist()) ],
                "choices": {
                    "yes": lambda user_id, db_session, curr_session, app: self.get_next_question_sad(user_id, app), #"personal_crisis_sad",
                    "no": "suggestions",
                },
                "protocols": {
                    "yes": [],
                    "no": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[19]],
                },
            },


            "personal_crisis_sad": {
                "model_prompt": [ np.random.choice(self.data["Sad - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question_sad(user_id, app), #"suggestions"
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [],
                },
            },

            #----------------------ANGER----------------------#

            "after_classification_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Was this caused by a specific event/s?"].dropna().tolist()) ],
                "choices": {
                    "yes": "event_is_recent_angry",
                    "no": "more_questions_angry", #change this. suggest protocols if no event or ask other questions?
                },
                "protocols": {"yes": [], "no": []},
            },

            "event_is_recent_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Was this caused by a recent or distant event (or events)?"].dropna().tolist()) ],
                "choices": {
                    "It was recent": "revisiting_recent_events_angry",
                    "It was distant": "revisiting_distant_events_angry",
                },
                "protocols": {"It was recent": [], "It was distant": []},
            },

            "revisiting_recent_events_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?"].dropna().tolist()) ],
                #+ "{} and found this reignited emotions of old events? ".format(
                #    self.PROTOCOL_TITLES[11]
                #),
                "choices": {
                    "yes": "more_questions_angry",
                    "no": "more_questions_angry",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[8]],
                    "no": [self.PROTOCOL_TITLES[11]],
                },
            },

            "revisiting_distant_events_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?"].dropna().tolist()) ],
                #+ "{} and found this reignited emotions of ".format(
                #    self.PROTOCOL_TITLES[6]
                #)
                #+ "old events?",
                "choices": {
                    "yes": "more_questions_angry",
                    "no": "more_questions_angry",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [self.PROTOCOL_TITLES[6]],
                },
            },

            "more_questions_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Thank you. Now I will ask some questions to understand your situation."].dropna().tolist()) ],
                "choices": {
                    "Okay": lambda user_id, db_session, curr_session, app: self.get_next_question_angry(user_id, app),
                    "I'd rather not": "suggestions",
                },
                "protocols": {
                    "Okay": [],
                    "I'd rather not": [],
                },
            },

            "displaying_antisocial_behaviour_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Have you strongly felt or expressed any of the following emotions towards someone:"].dropna().tolist()),
                "Envy, jealousy, greed, hatred, mistrust, "
                "malevolence, or revengefulness?"],
                "choices": {
                    "yes": "suggestions",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question_angry(user_id, app),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]],
                    "no": [],
                },
            },

            "internal_persecutor_saviour_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Do you believe that you should be the saviour of someone else?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "internal_persecutor_victim_angry",
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_victim_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Do you believe that you should be the saviour of someone else?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "internal_persecutor_controlling_angry",
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_controlling_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Do you feel that you are trying to control someone?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "internal_persecutor_accusing_angry",
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_accusing_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Are you always blaming and accusing yourself for when something goes wrong?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question_angry(user_id, app),
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
              },


            "rigid_thought_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - In previous conversations, have you considered other viewpoints presented?"].dropna().tolist()) ],
                "choices": {
                    "yes": lambda user_id, db_session, curr_session, app: self.get_next_question_angry(user_id, app),
                    "no": "suggestions",
                },
                "protocols": {
                    "yes": [],
                    "no": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[19]],
                },
            },


            "personal_crisis_angry": {
                "model_prompt": [ np.random.choice(self.data["Angry - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question_angry(user_id, app),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [],
                },
            },


            #-------------------FEAR/ANXIETY------------------#

            "after_classification_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Was this caused by a specific event/s?"].dropna().tolist()) ],
                "choices": {
                    "yes": "event_is_recent_scared",
                    "no": "more_questions_scared", #change this. suggest protocols if no event or ask other questions?
                },
                "protocols": {"yes": [], "no": []},
            },

            "event_is_recent_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Was this caused by a recent or distant event (or events)?"].dropna().tolist()) ],
                "choices": {
                    "It was recent": "revisiting_recent_events_scared",
                    "It was distant": "revisiting_distant_events_scared",
                },
                "protocols": {"It was recent": [], "It was distant": []},
            },

            "revisiting_recent_events_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?"].dropna().tolist()) ],
                #+ "{} and found this reignited emotions of old events? ".format(
                #    self.PROTOCOL_TITLES[11]
                #),
                "choices": {
                    "yes": "more_questions_scared",
                    "no": "more_questions_scared",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[8]],
                    "no": [self.PROTOCOL_TITLES[11]],
                },
            },

            "revisiting_distant_events_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?"].dropna().tolist()) ],
                #+ "{} and found this reignited emotions of ".format(
                #    self.PROTOCOL_TITLES[6]
                #)
                #+ "old events?",
                "choices": {
                    "yes": "more_questions_scared",
                    "no": "more_questions_scared",
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [self.PROTOCOL_TITLES[6]],
                },
            },

            "more_questions_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Thank you. Now I will ask some questions to understand your situation."].dropna().tolist()) ],
                "choices": {
                    "Okay": lambda user_id, db_session, curr_session, app: self.get_next_question_scared(user_id, app),
                    "I'd rather not": "suggestions",
                },
                "protocols": {
                    "Okay": [],
                    "I'd rather not": [],
                },
            },

            "displaying_antisocial_behaviour_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Have you strongly felt or expressed any of the following emotions towards someone:"].dropna().tolist()),
                "Envy, jealousy, greed, hatred, mistrust, "
                "malevolence, or revengefulness?"],
                "choices": {
                    "yes": "suggestions",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question_scared(user_id, app),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]],
                    "no": [],
                },
            },

            "internal_persecutor_saviour_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Do you believe that you should be the saviour of someone else?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "internal_persecutor_victim_scared",
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_victim_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Do you believe that you should be the saviour of someone else?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "internal_persecutor_controlling_scared",
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_controlling_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Do you feel that you are trying to control someone?"].dropna().tolist()) ],
                "choices": {
                "yes": "suggestions",
                "no": "internal_persecutor_accusing_scared",
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },

            "internal_persecutor_accusing_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Are you always blaming and accusing yourself for when something goes wrong?"].dropna().tolist()) ],
                "choices": {
                "yes": "suggestions", "no": lambda user_id, db_session, curr_session, app: self.get_next_question_scared(user_id, app),
                },
                "protocols": {"yes": self.INTERNAL_PERSECUTOR_PROTOCOLS, "no": []},
            },


            "rigid_thought_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - In previous conversations, have you considered other viewpoints presented?"].dropna().tolist()) ],
                "choices": {
                    "yes": lambda user_id, db_session, curr_session, app: self.get_next_question_scared(user_id, app),
                    "no": "suggestions",
                },
                "protocols": {
                    "yes": [],
                    "no": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[19]],
                },
            },


            "personal_crisis_scared": {
                "model_prompt": [ np.random.choice(self.data["Anxious - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": lambda user_id, db_session, curr_session, app: self.get_next_question_scared(user_id, app),
                },
                "protocols": {
                    "yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "no": [],
                },
            },

            #-------------------HAPPINESS------------------#

            "after_classification_happy": {
                "model_prompt": [ np.random.choice(self.data["Happy - That's Good! Let me recommend a protocol you can attempt."].dropna().tolist()) ],
                "choices": {
                    "Okay": "suggestions",
                    "No, thank you": "ending_prompt"
                },
                "protocols": {
                    "Okay": [self.PROTOCOL_TITLES[k] for k in self.positive_protocols],
                    "No, thank you": []
                },
            },


            #-------------------ALL EMOTIONS------------------#

            "suggestions": {
                "model_prompt": [ np.random.choice(self.data["All emotions - Here are my recommendations, please select the protocol that you would like to attempt"].dropna().tolist()) ],
                "choices": {
                    self.PROTOCOL_TITLES[k]: "trying_protocol"
                    for k in self.positive_protocols
                },
                "protocols": {
                    self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                    for k in self.positive_protocols
                },
            },


            "trying_protocol": {
                "model_prompt": [ np.random.choice(self.data["All emotions - Please try to go through this protocol now. When you finish, press 'continue'"].dropna().tolist()) ],
                "choices": {"continue": "user_found_useful"},
                "protocols": {"continue": []},
            },


            "user_found_useful": {
                "model_prompt": [ np.random.choice(self.data["All emotions - Do you feel better or worse after having taken this protocol?"].dropna().tolist()) ],
                "choices": {
                    "Better": "new_protocol_better",
                    "Worse": "new_protocol_worse",
                },
                "protocols": {"Better": [], "Worse": []},
            },


            "new_protocol_better": {
                "model_prompt": [ np.random.choice(self.data["All emotions - Would you like to attempt another protocol? (Patient feels better)"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "ending_prompt",
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },


            "new_protocol_worse": {
                "model_prompt": [ np.random.choice(self.data["All emotions - Would you like to attempt another protocol? (Patient feels worse)"].dropna().tolist()) ],
                "choices": {
                    "yes": "suggestions",
                    "no": "ending_prompt",
                },
                "protocols": {
                    "yes": [],
                    "no": [],
                },
            },


            "ending_prompt": {
                "model_prompt": [ np.random.choice(self.data["All emotions - Thank you for taking part. See you soon"].dropna().tolist()),
                                 "You have been disconnected. Refresh the page if you would like to start over." ],
                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []},

            },
        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())
        self.question_choices_sad = ["displaying_antisocial_behaviour_sad", "internal_persecutor_saviour_sad", "personal_crisis_sad", "rigid_thought_sad"]
        self.remaining_choices_sad = ["displaying_antisocial_behaviour_sad", "internal_persecutor_saviour_sad", "personal_crisis_sad", "rigid_thought_sad"]
        self.question_choices_angry = ["displaying_antisocial_behaviour_angry", "internal_persecutor_saviour_angry", "personal_crisis_angry", "rigid_thought_angry"]
        self.remaining_choices_angry = ["displaying_antisocial_behaviour_angry", "internal_persecutor_saviour_angry", "personal_crisis_angry", "rigid_thought_angry"]
        self.question_choices_scared = ["displaying_antisocial_behaviour_scared", "internal_persecutor_saviour_scared", "personal_crisis_scared", "rigid_thought_scared"]
        self.remaining_choices_scared = ["displaying_antisocial_behaviour_scared", "internal_persecutor_saviour_scared", "personal_crisis_scared", "rigid_thought_scared"]



    def get_suggestions(self, user_id, app):
        suggestions = []
        for curr_suggestions in list(self.suggestions[user_id]):
            suggestions.append(curr_suggestions[0])
        return suggestions

    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_scores(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def create_new_run(self, user_id, db_session, user_session):
        new_run = UserModelRun(session_id=user_session.id)
        db_session.add(new_run)
        db_session.commit()
        self.current_run_ids[user_id] = new_run.id
        return new_run

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {}

    def update_suggestions(self, user_id, protocols, app):

        # Check if user_id already has suggestions
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if type(protocols) != list:
            self.suggestions[user_id].append(deque([protocols]))
        else:
            self.suggestions[user_id].append(deque(protocols))

    # Takes next item in queue, or moves on to suggestions
    # if all have been checked


    def get_next_question_sad(self, user_id, app):
        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        if self.remaining_choices_sad == []:
            return "suggestions"
        elif current_choice == "more_questions_sad":
            selected_choice = np.random.choice(self.remaining_choices_sad)
            self.remaining_choices_sad.remove(selected_choice)
        elif current_choice in self.question_choices_sad:
            selected_choice = np.random.choice(self.remaining_choices_sad)
            self.remaining_choices_sad.remove(selected_choice)
        return selected_choice


    def get_next_question_angry(self, user_id, app):
        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        if self.remaining_choices_angry == []:
            return "suggestions"
        elif current_choice == "more_questions_angry":
            selected_choice = np.random.choice(self.remaining_choices_angry)
            self.remaining_choices_angry.remove(selected_choice)
        elif current_choice in self.question_choices_angry:
            selected_choice = np.random.choice(self.remaining_choices_angry)
            self.remaining_choices_angry.remove(selected_choice)
        return selected_choice


    def get_next_question_scared(self, user_id, app):
        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        if self.remaining_choices_scared == []:
            return "suggestions"
        elif current_choice == "more_questions_scared":
            selected_choice = np.random.choice(self.remaining_choices_scared)
            self.remaining_choices_scared.remove(selected_choice)
        elif current_choice in self.question_choices_scared:
            selected_choice = np.random.choice(self.remaining_choices_scared)
            self.remaining_choices_scared.remove(selected_choice)
        return selected_choice


    def add_to_reordered_protocols(self, user_id, next_protocol):
        self.reordered_protocol_questions[user_id].append(next_protocol)

    def add_to_next_protocols(self, next_protocols):
        self.protocols_to_suggest.append(deque(next_protocols))

    def clear_suggested_protocols(self):
        self.protocols_to_suggest = []

    # NOTE: this is not currently used, but can be integrated to support
    # positive protocol suggestions (to avoid recent protocols).
    # You would need to add it in when a user's emotion is positive
    # and they have chosen a protocol.

    def add_to_recent_protocols(self, recent_protocol):
        if len(self.recent_protocols) == self.recent_protocols.maxlen:
            # Removes oldest protocol
            self.recent_protocols.popleft()
        self.recent_protocols.append(recent_protocol)

    def calculate_score(self, user_input, for_keywords, against_keywords):
        # Punctuation removal/tokenisation

        score = 0
        for word in user_input:
            if word in for_keywords:
                score += 1

            elif word in against_keywords:
                score -= 1

        return score


    def get_model_prompt_guess_emotion(self, user_id):
        return "Based on what you have said, I believe you are: {}. Is this correct?".format(
            self.guess_emotion_predictions[user_id]
        )

    def determine_next_prompt_opening(self, user_id, app, db_session):
        emotion_scores = self.calculate_emotion_scores(user_id, app)

        (
            happy_score,
            sad_score,
            angry_score,
            scared_score,
        ) = emotion_scores

        emotion_detected = False
        for emotion_score in emotion_scores:
            if emotion_score > 0:
                emotion_detected = True
                break

        current_run = UserModelRun.query.filter_by(
            id=self.current_run_ids[user_id]
        ).first()
        current_run.emotion_happy_score = happy_score
        current_run.emotion_sad_score = sad_score
        current_run.emotion_angry_score = angry_score
        current_run.emotion_scared_score = scared_score
        db_session.commit()

        if emotion_detected:
            max_pos = np.argmax(emotion_scores)

            # We stop when we find the first relevant case.
            if max_pos == 0:
                self.guess_emotion_predictions[user_id] = "happy"

            elif max_pos == 1:
                self.guess_emotion_predictions[user_id] = "sad"

            elif max_pos == 2:
                self.guess_emotion_predictions[user_id] = "angry"

            elif max_pos == 3:
                self.guess_emotion_predictions[user_id] = "scared"

            return "guess_emotion"
        else:
            return "check_emotion"

    def determine_next_prompt_new_protocol(self, user_id, app):
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if len(self.suggestions[user_id]) > 0:
            return "suggestions"

        return "opening_prompt"

    def extract_for_and_against_keywords(
        self, starting_for_keywords, starting_against_keywords
    ):
        for_keywords = starting_for_keywords
        against_keywords = starting_against_keywords
        for i in range(len(starting_for_keywords)):
            word = starting_for_keywords[i]
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    # Adds lemmas to for_keywords
                    for_keywords.append(lemma.name())
                    if lemma.antonyms():
                        for antonym in lemma.antonyms():
                            against_keywords.append(antonym.name())

        return set(for_keywords), set(against_keywords)

    def detect_happy(self, user_input):
        # Note: synonym included for for_keywords is OK so this
        # will overlap with detect_neutral (even with adding to against_keywords)
        # but it is hard to detect a neutral emotion with keywords in any case.
        starting_for_keywords = [
            "good",
            "happy",
            "fine",
            "well",
            "delighted",
            "great",
            "positive",
            "decent",
        ]
        starting_against_keywords = [
            "sad",
            "unhappy",
            "angry",
            "scared",
            "anxious",
            "upset",
            "ok",
        ]

        # Keywords for detecting and not detecting the behaviour, respectively
        for_keywords, against_keywords = self.extract_for_and_against_keywords(
            starting_for_keywords, starting_against_keywords
        )

        return self.calculate_score(user_input, for_keywords, against_keywords)


    def detect_sad(self, user_input):
        starting_for_keywords = ["bad", "sad", "unhappy", "depressed"]
        starting_against_keywords = [
            "neutral",
            "angry",
            "scared",
            "anxious",
            "happy",
            "worried",
            "not",
        ]

        # Keywords for detecting and not detecting the behaviour, respectively
        for_keywords, against_keywords = self.extract_for_and_against_keywords(
            starting_for_keywords, starting_against_keywords
        )

        return self.calculate_score(user_input, for_keywords, against_keywords)


    def detect_angry(self, user_input):
        starting_for_keywords = ["angry", "furious", "rage"]
        starting_against_keywords = [
            "neutral",
            "upset",
            "depressed",
            "scared",
            "anxious",
            "worried",
            "happy",
        ]

        # Keywords for detecting and not detecting the behaviour, respectively
        for_keywords, against_keywords = self.extract_for_and_against_keywords(
            starting_for_keywords, starting_against_keywords
        )

        return self.calculate_score(user_input, for_keywords, against_keywords)


    def detect_scared(self, user_input):
        starting_for_keywords = ["scared", "afraid"]
        starting_against_keywords = ["neutral", "upset", "depressed", "happy"]

        # Keywords for detecting and not detecting the behaviour, respectively
        for_keywords, against_keywords = self.extract_for_and_against_keywords(
            starting_for_keywords, starting_against_keywords
        )

        return self.calculate_score(user_input, for_keywords, against_keywords)




    def determine_positive_protocols(self, user_id, app):
        protocol_counts = {}
        total_count = 0

        for protocol in self.positive_protocols:
            count = Protocol.query.filter_by(protocol_chosen=protocol).count()
            protocol_counts[protocol] = count
            total_count += count

        # for protocol in counts:
        if total_count > 10:
            first_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[first_item]

            second_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[second_item]

            third_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[third_item]
        else:
            # CASE: < 10 protocols undertaken in total, so randomness introduced
            # to avoid lowest 3 being recommended repeatedly.
            # Gives number of next protocol to be suggested
            first_item = np.random.choice(
                list(set(self.positive_protocols) - set(self.recent_protocols))
            )
            second_item = np.random.choice(
                list(
                    set(self.positive_protocols)
                    - set(self.recent_protocols)
                    - set([first_item])
                )
            )
            third_item = np.random.choice(
                list(
                    set(self.positive_protocols)
                    - set(self.recent_protocols)
                    - set([first_item, second_item])
                )
            )

        return [
            self.PROTOCOL_TITLES[first_item],
            self.PROTOCOL_TITLES[second_item],
            self.PROTOCOL_TITLES[third_item],
        ]

    def determine_protocols_keyword_classifiers(
        self, user_id, db_session, curr_session, app
    ):

        # We add "suggestions" first, and in the event there are any left over we use those, otherwise we divert past it.
        self.add_to_reordered_protocols(user_id, "suggestions")

        # Default case: user should review protocols 13 and 14.
        self.add_to_next_protocols([self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]])
        return self.get_next_protocol_question(user_id, app)



    def calculate_emotion_scores(self, user_id, app):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["opening_prompt"]
        except:  # noqa
            user_response = ""
        user_input = user_response.lower()
        tokenised_input = re.findall(r"[\w']+", user_input)

        scores = [
            self.detect_happy(tokenised_input),
            self.detect_sad(tokenised_input),
            self.detect_angry(tokenised_input),
            self.detect_scared(tokenised_input),
        ]

        return scores


    def update_conversation(self, user_id, new_dialogue, db_session, app):
        try:
            session_id = self.user_choices[user_id]["current_session_id"]
            curr_session = UserModelSession.query.filter_by(id=session_id).first()
            if curr_session.conversation is None:
                curr_session.conversation = "" + new_dialogue
            else:
                curr_session.conversation = curr_session.conversation + new_dialogue
            curr_session.last_updated = datetime.datetime.utcnow()
            db_session.commit()
        except KeyError:
            curr_session = UserModelSession(
                user_id=user_id,
                conversation=new_dialogue,
                last_updated=datetime.datetime.utcnow(),
            )

            db_session.add(curr_session)
            db_session.commit()
            self.user_choices[user_id]["current_session_id"] = curr_session.id


    def save_current_choice(
        self, user_id, input_type, user_choice, user_session, db_session, app
    ):
        # Set up dictionary if not set up already
        # with Session() as session:

        try:
            self.user_choices[user_id]
        except KeyError:
            self.user_choices[user_id] = {}

        # Define default choice if not already set
        try:
            current_choice = self.user_choices[user_id]["choices_made"][
                "current_choice"
            ]
        except KeyError:
            current_choice = self.QUESTION_KEYS[0]

        try:
            self.user_choices[user_id]["choices_made"]
        except KeyError:
            self.user_choices[user_id]["choices_made"] = {}

        if current_choice == "opening_prompt":
            self.clear_suggestions(user_id)
            self.user_choices[user_id]["choices_made"] = {}
            self.create_new_run(user_id, db_session, user_session)

        # Save current choice
        self.user_choices[user_id]["choices_made"]["current_choice"] = current_choice
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice

        curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        # prompt_to_use = curr_prompt
        if callable(curr_prompt):
            curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        #removed stuff here

        else:
            self.update_conversation(
                user_id,
                "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
                db_session,
                app,
            )

        # Case: update suggestions for next attempt by removing relevant one
        if (
            current_choice == "suggestions"
        ):
            if current_choice == "after_classification_happy":
                self.user_emotions[user_id] = "happy"

            elif current_choice == "after_classification_sad":
                self.user_emotions[user_id] = "sad"

            elif current_choice == "after_classification_angry":
                self.user_emotions[user_id] = "angry"

            elif current_choice == "after_classification_scared":
                self.user_emotions[user_id] = "scared"

            #if current_choice == "after_classification_happy":
            #    self.user_emotions[user_id] = "happy"

            # PRE: user_choice is a string representing a number from 1-20,
            # or the title for the corresponding protocol

            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
            except KeyError:
                current_protocol = int(user_choice)

            protocol_chosen = Protocol(
                protocol_chosen=current_protocol,
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )
            db_session.add(protocol_chosen)
            db_session.commit()
            self.current_protocol_ids[user_id] = protocol_chosen.id

            for i in range(len(self.suggestions[user_id])):
                curr_protocols = self.suggestions[user_id][i]
                if curr_protocols[0] == self.PROTOCOL_TITLES[current_protocol]:
                    curr_protocols.popleft()
                    if len(curr_protocols) == 0:
                        self.suggestions[user_id].pop(i)
                    break

        # PRE: User choice is string in ["Better", "Worse"]
        elif current_choice == "user_found_useful":
            current_protocol = Protocol.query.filter_by(
                id=self.current_protocol_ids[user_id]
            ).first()
            current_protocol.protocol_was_useful = user_choice
            db_session.commit()

        if current_choice == "guess_emotion":
            option_chosen = user_choice + " ({})".format(
                self.guess_emotion_predictions[user_id]
            )
        else:
            option_chosen = user_choice
        choice_made = Choice(
            choice_desc=current_choice,
            option_chosen=option_chosen,
            user_id=user_id,
            session_id=user_session.id,
            run_id=self.current_run_ids[user_id],
        )
        db_session.add(choice_made)
        db_session.commit()

        return choice_made

    def determine_next_choice(
        self, user_id, input_type, user_choice, db_session, user_session, app
    ):
        # Find relevant user info by using user_id as key in dict.
        #
        # Then using the current choice and user input, we determine what the next
        # choice is and return this as the output.

        # Some edge cases to consider based on the different types of each field:
        # May need to return list of model responses. For next protocol, may need
        # to call function if callable.

        # If we cannot find the specific choice (or if None etc.) can set user_choice
        # to "any".

        # PRE: Will be defined by save_current_choice if it did not already exist.
        # (so cannot be None)

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]
        current_protocols = self.QUESTIONS[current_choice]["protocols"]
        if input_type != "open_text":
            if (
                current_choice != "suggestions"
                and current_choice != "event_is_recent_sad"
                and current_choice != "event_is_recent_angry"
                and current_choice != "event_is_recent_scared"
                and current_choice != "more_questions_sad"
                and current_choice != "more_questions_angry"
                and current_choice != "more_questions_scared"
                and current_choice != "after_classification_happy"
                and current_choice != "user_found_useful"
            ):
                user_choice = user_choice.lower()

            if (
                current_choice == "suggestions"
            ):
                try:
                    current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
                except KeyError:
                    current_protocol = int(user_choice)
                protocol_choice = self.PROTOCOL_TITLES[current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            elif current_choice == "check_emotion":
                if user_choice == "sad":
                    next_choice = current_choice_for_question["sad"]
                    protocols_chosen = current_protocols["sad"]
                elif user_choice == "angry":
                    next_choice = current_choice_for_question["angry"]
                    protocols_chosen = current_protocols["angry"]
                elif user_choice == "scared":
                    next_choice = current_choice_for_question["scared"]
                    protocols_chosen = current_protocols["scared"]
                else:
                    next_choice = current_choice_for_question["happy"]
                    protocols_chosen = current_protocols["happy"]
            else:
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        else:
            next_choice = current_choice_for_question["open_text"]
            protocols_chosen = current_protocols["open_text"]

        if callable(next_choice):
            next_choice = next_choice(user_id, db_session, user_session, app)

        if current_choice == "guess_emotion" and user_choice.lower() == "yes":
            if self.guess_emotion_predictions[user_id] == "sad":
                next_choice = next_choice["sad"]
            elif self.guess_emotion_predictions[user_id] == "angry":
                next_choice = next_choice["angry"]
            elif self.guess_emotion_predictions[user_id] == "scared":
                next_choice = next_choice["scared"]
            else:
                next_choice = next_choice["happy"]

        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(user_id, db_session, user_session, app)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]
        if callable(next_prompt):
            next_prompt = next_prompt(user_id, db_session, user_session, app)
        if (
            len(protocols_chosen) > 0
            and current_choice != "suggestions"
        ):
            self.update_suggestions(user_id, protocols_chosen, app)

        # Case: new suggestions being created after first protocol attempted
        if next_choice == "opening_prompt":
            self.clear_suggestions(user_id)
            self.clear_emotion_scores(user_id)
            self.create_new_run(user_id, db_session, user_session)

        if next_choice == "suggestions":
            next_choices = self.get_suggestions(user_id, app)

        else:
            next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())
        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice
        return {"model_prompt": next_prompt, "choices": next_choices}
