import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load spaCy NLP model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "en_core_web_sm/en_core_web_sm-3.8.0")
nlp = spacy.load(MODEL_PATH)

# Training Data for Intent Classification
# Training Data for Intent Classification
X_train = [
    "I want to book a room",
    "Can I reserve a hotel room?",
    "I need a room for 2 guests",
    "My name is John Doe",
    "I am Sarah",
    "Check-in on January 5, check-out on January 10",
    "We are 3 people staying",
    "How much does it cost?",
    "Do you have breakfast included?",
    "Is breakfast available?",
    "Yes, I want breakfast",
    "No, I don’t want breakfast",
    "I will pay with Visa",
    "I want a deluxe room",
    "Yes, confirm my booking",
    "Yes, proceed with booking",
    "Proceed with my booking",
    "I want to confirm the reservation"
]

y_train = [
    "book_room",
    "book_room",
    "provide_guests",
    "provide_name",
    "provide_name",
    "provide_dates",
    "provide_guests",
    "ask_price",
    "ask_breakfast",
    "ask_breakfast",
    "ask_breakfast",
    "ask_breakfast",
    "ask_payment",
    "choose_room_type",
    "confirm_booking",
    "proceed_with_booking",
    "proceed_with_booking",
    "confirm_booking"
]

# Train Model
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# Function to classify user intent
def classify_intent(user_input):
    X_test_vectors = vectorizer.transform([user_input])
    return model.predict(X_test_vectors)[0]

# Function to extract user entities (Name, Dates, Number of Guests)
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {ent.label_: ent.text for ent in doc.ents}

    # Handle multiple dates (Check-in & Check-out)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    if len(dates) == 2:
        entities["CHECK_IN"] = dates[0]
        entities["CHECK_OUT"] = dates[1]
    elif len(dates) == 1:
        entities["CHECK_IN"] = dates[0]
        entities["CHECK_OUT"] = None  

    return entities

# Function to process chatbot response with session memory
def chatbot_response(session_id, user_input, conversation_memory):
    """Handles conversation flow and memory tracking for each user session."""

    # Ensure user memory exists and initialize required fields
    if "user_data" not in conversation_memory:
        conversation_memory["user_data"] = {}

    user_data = conversation_memory["user_data"]
    required_keys = ["name", "check_in", "check_out", "guests", "breakfast", "payment", "room_type", "confirmed"]
    for key in required_keys:
        user_data.setdefault(key, None)

    # Classify user intent
    intent = classify_intent(user_input)
    entities = extract_entities(user_input)

    # Store extracted entities in memory
    if intent == "provide_name" and "PERSON" in entities:
        user_data["name"] = entities["PERSON"]

    elif intent == "provide_dates":
        user_data["check_in"] = entities.get("CHECK_IN", user_data["check_in"])
        user_data["check_out"] = entities.get("CHECK_OUT", user_data["check_out"])

    elif intent == "provide_guests" and "CARDINAL" in entities:
        user_data["guests"] = entities["CARDINAL"]

    elif intent == "ask_payment":
        user_data["payment"] = "Visa" if "visa" in user_input.lower() else "Mastercard" if "mastercard" in user_input.lower() else "PayPal"

    elif intent == "choose_room_type":
        user_data["room_type"] = "deluxe" if "deluxe" in user_input.lower() else "standard"

    # Ensure the bot does not reset the name
    if user_data["name"]:
        user_greeting = f"Great, {user_data['name']}! "
    else:
        user_greeting = ""

    # Print Updated Memory
    print("DEBUG: Updated Conversation Memory:", conversation_memory)

    # Handle responses based on conversation flow
    if intent == "book_room":
        # Reset memory when starting a new booking session
        print("DEBUG: Resetting conversation memory for a new session.")
        conversation_memory["user_data"] = {
            "name": None,
            "check_in": None,
            "check_out": None,
            "guests": None,
            "breakfast": None,
            "payment": None,
            "room_type": None,
            "confirmed": None
        }

        return "Welcome to Grand Vista Hotel! Can you provide your full name?"

    elif intent == "provide_name":
        return f"Nice to meet you, {user_data['name']}! When would you like to check in and check out?"

    elif intent == "provide_dates":
        return f"Got it! Your stay is from {user_data['check_in']} to {user_data['check_out']}. How many guests will be staying?"

    elif intent == "provide_guests":
        return f"Okay, {user_data['guests']} guests will be staying. Would you like to include breakfast?"

    elif intent == "ask_breakfast":
        user_data["breakfast"] = "yes" if "yes" in user_input.lower() else "no"
        return user_greeting + "Breakfast is available for an additional charge of $10 per guest. What type of room would you like to book - standard or deluxe?"

    elif intent == "ask_payment":
        return user_greeting + f"Your payment method is set to {user_data['payment']}. Do you want to proceed with your booking?"

    elif intent == "ask_price":
        return "The price depends on the room type and duration. Do you want a standard or deluxe room?"

    elif intent == "choose_room_type":
        return user_greeting + f"You have chosen a {user_data['room_type']} room. How would you like to proceed with payment?"
    
    elif intent == "proceed_with_booking":
        return f"Great, {user_data['name']}! Your booking is almost complete. You have chosen a {user_data['room_type']} room at ${'150' if user_data['room_type']=='deluxe' else '100'} per night. Your payment method is {user_data['payment']}. efore confirming, please note that you can cancel within 24 hours of booking for a full refund Would you still like to confirm your reservation?"

    elif intent == "confirm_booking":
        if None in (user_data["name"], user_data["check_in"], user_data["check_out"], user_data["guests"], user_data["payment"]):
            return "Some details are missing! Please provide your name, check-in/check-out dates, number of guests, and payment method."
        user_data["confirmed"] = True
        return f"""Your booking is confirmed! Here is your reservation summary:

            📌 Booking Summary:\n
            - Name: {user_data['name']}\n
            - Check-in: {user_data['check_in']}\n
            - Check-out: {user_data['check_out']}\n
            - Guests: {user_data['guests']}\n
            - Room Type: {user_data['room_type'].capitalize()}\n
            - Breakfast: {'Yes' if user_data['breakfast'] == 'yes' else 'No'}\n
            - Price per Night: ${'150' if user_data['room_type']=='deluxe' else '100'}\n
            - Payment Method: {user_data['payment']}\n
            
            ✅ Thank you for choosing Grand Vista Hotel, {user_data['name']}! You will receive an email with your booking details.
            """
        

    elif intent == "ask_cancellation":
        return "You can cancel within 24 hours of booking for a full refund. Do you still want to confirm your booking?"

    return "I'm sorry, I didn't understand that. Can you please rephrase?"