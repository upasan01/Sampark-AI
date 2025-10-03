import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from google.cloud import translate_v2 as translate # Used to simulate accurate external translation

# --- Configuration ---
# Ensure your GEMINI_API_KEY is set in your environment
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# --- 1. Core AI Model (Simulates Sampark AI's understanding) ---
def initialize_ai_model():
    """Initializes the Gemini model for the chatbot."""
    # Using gemini-2.5-flash for fast, high-quality multilingual performance
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    return model

# --- 2. External Translation Service (Simulates the multilingual module) ---
def translate_text(text, target_language_code):
    """Translates text using Google Cloud Translate API."""
    try:
        translate_client = translate.Client()
        # The target language code is the key feature for multilingual support (e.g., 'hi' for Hindi)
        result = translate_client.translate(text, target_language=target_language_code)
        return result['translatedText']
    except Exception as e:
        # Fallback to English if translation fails (e.g., credentials missing)
        print(f"Warning: Translation failed ({e}). Falling back to English response.")
        return text

# --- 3. The Sampark AI Chat Engine ---
def sampark_ai_chat(user_prompt: str, language_code: str = 'en'):
    """
    Handles the user request, gets the AI response, and translates it back.

    :param user_prompt: The user's input text (can be any language).
    :param language_code: The target language code for the response (e.g., 'hi', 'es').
    """
    model = initialize_ai_model()

    # Define the system prompt to instruct the AI's role and tone
    system_instruction = (
        "You are Sampark-AI, a helpful, polite, and official customer support assistant "
        "for a government agency. Keep your answers concise, accurate, and professional. "
        "Always respond in clear, formal English, as your final output will be translated."
    )

    # The LangChain message history
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_prompt)
    ]

    print(f"\n[USER PROMPT in {language_code.upper()}]: {user_prompt}")

    # 1. Get the AI's response in English (the model's native processing language)
    try:
        ai_response_en = model.invoke(messages).content
        print(f"[AI RESPONSE (English)]: {ai_response_en}")
    except Exception as e:
        return f"AI Error: Could not generate response. Check your GEMINI_API_KEY. Details: {e}"

    # 2. Translate the English response to the desired language code
    if language_code.lower() != 'en':
        translated_response = translate_text(ai_response_en, language_code)
        print(f"[AI RESPONSE ({language_code.upper()})]: {translated_response}")
        return translated_response
    else:
        # If the user requested English, return the English response directly
        return ai_response_en


# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: English Query
    print("--- Example 1: English Query ---")
    sampark_ai_chat(
        user_prompt="What is the process for applying for a new identity card?",
        language_code='en'
    )

    # Example 2: Hindi Query and Response
    print("\n" + "="*50 + "\n")
    print("--- Example 2: Hindi Query and Response ('hi') ---")
    sampark_ai_chat(
        user_prompt="मैं अपनी बेटी के लिए जन्म प्रमाण पत्र कैसे प्राप्त कर सकता हूं?",
        language_code='hi' # Hindi
    )

    # Example 3: Spanish Query and Response
    print("\n" + "="*50 + "\n")
    print("--- Example 3: Spanish Query and Response ('es') ---")
    sampark_ai_chat(
        user_prompt="Necesito saber el horario de la oficina de impuestos.",
        language_code='es' # Spanish
    )
