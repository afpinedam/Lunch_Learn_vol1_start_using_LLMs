import math
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
# Read the README.md file for more information about the API
api_key = os.getenv("OPENAI_API_KEY")
# Ensure API key is defined, otherwise raise an error
assert api_key is not None, "Define OPENAI_API_KEY in the .env file"

# Create OpenAI client instance with the API key
client = OpenAI(api_key=api_key)

print("API key loaded successfully")


# =============================
# MODEL CONFIGURATION
# =============================
# Dictionary containing available language models and their pricing
# Format: "key": {"name": "model_name", "price per 1M tokens": price_per_million_tokens}
models = {
    "1": {"name": "gpt-4o-mini", "price per 1M tokens": 0.15},
    "2": {"name": "gpt-4.1-mini", "price per 1M tokens": 0.40},
    "3": {"name": "gpt-4.1", "price per 1M tokens": 2.00},
    "4": {"name": "gpt-5-nano", "price per 1M tokens": 0.05},
    "5": {"name": "gpt-5-mini", "price per 1M tokens": 0.25},
}

# =============================
# FUNCTIONS
# =============================

def estimate_tokens(text):
    """
    Estimates the number of tokens in an input text.

    Uses a simple approximation where 1 token ≈ 4 characters.
    This is a rough estimation; actual token count may vary by model.

    Args:
        text (str): Input text provided by the user.

    Returns:
        int: Estimated number of tokens in the text.
    """
    # Common approximation: 1 token ≈ 4 characters
    # Calculate by dividing text length by 4 and rounding up
    return math.ceil(len(text) / 4)


def estimate_cost(tokens, price_per_1m):
    """
    Calculates the estimated input cost based on the number of tokens.

    The calculation is based on the price per million tokens defined for each model.

    Args:
        tokens (int): Estimated number of input tokens.
        price_per_1m (float): Model price per 1 million tokens.

    Returns:
        float: Estimated cost in USD for the input.
    """
    # Formula: (tokens / 1,000,000) * price_per_million = cost
    return (tokens / 1000000) * price_per_1m


def select_model(tokens):
    """
    Allows the user to select an available language model.

    Displays a list of available models with the estimated cost of the query
    based on the input tokens.

    Args:
        tokens (int): Estimated number of tokens in the query.

    Returns:
        dict: Dictionary containing the configuration of the selected model.
    """
    print("\nSelect a model:\n")
    # Loop through available models and display them with estimated cost
    for key, model in models.items():
        # Calculate cost for this specific model
        cost = estimate_cost(tokens, model["price per 1M tokens"])
        # Display model option with its estimated cost
        print(f"{key}. {model['name']} → Estimated cost: ${cost:.6f}")

    # Keep asking until user selects a valid model
    while True:
        try:
            # Get user's model selection
            choice = input("\nOption: ")

            # Check if the selected option is valid (exists in models dictionary)
            if choice in models:
                model = models[choice]
                print(f"Selected model: {model['name']}")
                return model
            else:
                print("Invalid model selection")

        except ValueError:
            print("You must enter a valid number.")
                

def select_max_tokens():
    """
    Requests the maximum number of output tokens from the user.

    If the user presses Enter without entering a value, output is unlimited (None).
    This controls how long the model's response can be.

    Returns:
        int | None: Maximum number of output tokens, or None if no limit is specified.
    """
    # Keep asking until user provides valid input
    while True:
        # Prompt user for max tokens (Enter key = unlimited)
        value = input("\nMaximum output tokens (Press Enter = unlimited): ")

        # If user just presses Enter, return None (unlimited tokens)
        if value.strip() == "":
            print("Maximum output tokens: Unlimited")
            return None

        try:
            # Convert input string to integer
            value_int = int(value)

            # Ensure value is positive
            if value_int > 0:
                print(f"Maximum output tokens: {value_int}")
                return value_int
            else:
                print("The value must be greater than 0.")

        except ValueError:
            print("The value is not a valid integer.")


def select_temperature():
    """
    Allows the user to select the creativity level of the model.

    Temperature controls randomness in responses:
    - Lower values (closer to 0) = more deterministic/precise
    - Higher values (closer to 1 or above) = more creative/random

    Returns:
        float: Temperature value selected for text generation.
    """
    print("\nCreativity level:")
    print("1. Very precise (0.2)")
    print("2. Balanced (0.5)")
    print("3. Creative (0.8)")
    print("4. Very creative (1.0)")

    # Map user's numeric choice to temperature value
    mapping = {
        1: 0.2,
        2: 0.5,
        3: 0.8,
        4: 1.0
    }

    # Keep asking until user selects a valid option
    while True:
        # Get user's creativity level selection
        choice = input("Option: ")

        try:
            # Convert input to integer
            value_int = int(choice)

            # Check if selection is valid (exists in mapping)
            if value_int in mapping:
                temperature = mapping[value_int]
                print(f"Temperature used: {temperature}")
                return temperature
            else:
                print("Invalid option. Choose between 1 and 4.")

        except ValueError:
            print("The value is not a valid integer.")
    
    


def query_model(prompt, model_name, max_tokens, temperature):
    """
    Sends a query to the language model and returns the generated response.

    This function makes an API call to OpenAI with the specified parameters,
    measures latency, and displays token usage information.

    Args:
        prompt (str): User's input text/question.
        model_name (str): Name of the model to use for the query.
        max_tokens (int | None): Maximum limit of output tokens.
        temperature (float): Creativity level of the model (0.0 to 2.0).

    Returns:
        str: Response generated by the model or error message if the request fails.
    """
    # Start timer to measure latency
    start = time.perf_counter()
    
    # Send request to OpenAI API with specified parameters
    response = client.responses.create(
        model=model_name,
        input=prompt,
        max_output_tokens=max_tokens,
        temperature=temperature
    )
    
    # End timer to calculate latency
    end = time.perf_counter()

    # Calculate time elapsed during API request
    latency = end - start

    # Display token usage and latency information
    print(f"Number of input tokens: {response.usage.input_tokens}")
    print(f"Number of output tokens: {response.usage.output_tokens}")
    print(f"Processing time (latency) in seconds: {latency}")
    
    # Extract and return the generated text from the response
    return response.output[0].content[0].text



# =============================
# MAIN PROGRAM FLOW
# =============================

def main():
    """
    Main function that executes the interactive program flow.

    Allows the user to:
    - Ask a question or exit the program
    - See the estimated number of tokens
    - Select a language model
    - Define the output tokens limit
    - Choose the creativity level
    - Get a response generated by the model

    The program runs in a loop until the user chooses to exit.
    """
    
    # Main program loop - runs continuously until user exits
    while True:
        # Display menu options
        print("Press 1 to ask a question")
        print("Press 2 to exit")
        # Get user's menu selection
        menu = input()

        # Handle user's menu selection
        if menu == "1":
            
            # Get the user's question/prompt
            prompt = input("Enter your question:\n")
            
            # Validate that the prompt is not empty
            if not prompt.strip():
                print("The question cannot be empty.\n")
                continue
            
            # Step 1: Estimate tokens for the input prompt
            tokens = estimate_tokens(prompt)
            print(f"\nEstimated tokens: {tokens}")

            # Step 2: Let user select a model
            model = select_model(tokens)
                
            # Step 3: Let user define maximum output tokens
            max_tokens = select_max_tokens()

            # Step 4: Let user select creativity level (temperature)
            temperature = select_temperature()

            # Step 5: Query the model with all selected parameters
            print("\nQuerying model...\n")

            answer = query_model(
                prompt,
                model["name"],
                max_tokens,
                temperature
            )

            # Display the model's response
            print("Response:\n")
            print(answer)

        elif menu == "2":
            # Exit message when user chooses option 2
            print("Goodbye!")
            break
        
        else:
            # Handle invalid menu selection
            print("Invalid option")
        


# Entry point of the program
# This ensures main() only runs when the script is executed directly,
# not when it's imported as a module
if __name__ == "__main__":
    main()
