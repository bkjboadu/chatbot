# 1️⃣ User: I want to book a room
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "I want to book a room"}'

# 2️⃣ User: My name is Michael Johnson
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "My name is Michael Johnson"}'

# 3️⃣ User: Check-in on January 5, check-out on January 10
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "Check-in on January 5, check-out on January 10"}'

# 4️⃣ User: We are 3 guests staying
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "We are 3 guests staying"}'

# 5️⃣ User: Yes, please include breakfast
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "Yes, please include breakfast"}'

# 6️⃣ User: Yes, proceed with booking
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "Yes, proceed with booking"}'

# 7️⃣ User: I will pay with Visa
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "I will pay with Visa"}'

# 8️⃣ User: How much does it cost?
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "How much does it cost?"}'

# 9️⃣ User: I want a deluxe room
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "I want a deluxe room"}'

# 🔟 User: Yes, confirm my booking
curl -X POST http://127.0.0.1:8000/chatbot/chat/ -H "Content-Type: application/json" -d '{"message": "Yes, confirm my booking"}'