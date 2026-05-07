import math
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Obtener API key
# Leer el archivo README.md para más información de la API
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "Define OPENAI_API_KEY en el .env"

# Crear cliente
client = OpenAI(api_key=api_key)

print("API key cargada correctamente")


# -----------------------------
# CONFIGURACIÓN DE MODELOS
# -----------------------------
models = {
    "1": {"name": "gpt-4o-mini", "precio por 1M de tokens": 0.15},
    "2": {"name": "gpt-4.1-mini", "precio por 1M de tokens": 0.40},
    "3": {"name": "gpt-4.1", "precio por 1M de tokens": 2.00},
    "4": {"name": "gpt-5-nano", "precio por 1M de tokens": 0.05},
    "5": {"name": "gpt-5-mini", "precio por 1M de tokens": 0.25},
    "6": {"name": "gpt-3.5-turbo", "precio por 1M de tokens": 0.25},
}

# Precio por 1M tokens de OUTPUT (USD)
output_pricing = {
    "gpt-4o-mini": 0.60,
    "gpt-4.1-mini": 1.60,
    "gpt-4.1": 8.00,
    "gpt-5-nano": 0.20,
    "gpt-5-mini": 1.00,
    "gpt-3.5-turbo": 0.50,
}
# -----------------------------
# FUNCIONES
# -----------------------------

def estimate_tokens(text):
    """
    Estima el número de tokens de un texto de entrada.

    Utiliza una aproximación simple donde 1 token ≈ 4 caracteres.

    Args:
        text (str): Texto de entrada proporcionado por el usuario.

    Returns:
        int: Número estimado de tokens del texto.
    """
    # Aproximación común: 1 token ≈ 4 caracteres
    return math.ceil(len(text) / 4)


def estimate_cost(tokens, price_per_1m):
    """
    Calcula el coste estimado de entrada basado en el número de tokens.

    El cálculo se basa en el precio por millón de tokens definido para cada modelo.

    Args:
        tokens (int): Número estimado de tokens de entrada.
        price_per_1M (float): Precio del modelo por 1 millón de tokens.

    Returns:
        float: Coste estimado en USD de la entrada.
    """
    return (tokens / 1000000) * price_per_1m


def select_model(tokens):
    """
    Permite al usuario seleccionar un modelo de lenguaje disponible.

    Muestra una lista de modelos junto con el coste estimado de la consulta
    basado en los tokens de entrada.

    Args:
        tokens (int): Número estimado de tokens de la consulta.

    Returns:
        dict: Diccionario con la configuración del modelo seleccionado.
    """
    print("\nSelecciona un modelo:\n")
    for key, model in models.items():
         cost = estimate_cost(tokens, model["precio por 1M de tokens"])
         print(f"{key}. {model['name']} → Coste estimado: ${cost:.6f}")

    while True:
        try:
            choice = input("\nOpción: ")

            if choice in models:
                model = models[choice]
                print(f"Modelo elegido: {model['name']}")
                return model
            else:
                print("Modelo no válido")

        except ValueError:
            print("Debes introducir un número válido.")
                

def select_max_tokens():
    """
    Solicita al usuario el número máximo de tokens de salida.

    Si el usuario no introduce ningún valor, se considera ilimitado (None).

    Returns:
        int | None: Número máximo de tokens de salida o None si no se especifica límite.
    """
    while True:
        value = input("\nMáximo de tokens de salida (Enter = ilimitado): ")

        if value.strip() == "":
            print("Máxima cantidad de tokens de salida: Ilimitado")
            return None

        try:
            value_int = int(value)

            if value_int > 0:
                print(f"Máxima cantidad de tokens de salida: {value_int}")
                return value_int
            else:
                print("El valor debe ser mayor que 0.")

        except ValueError:
            print("El valor no es un número entero.")


def select_temperature():
    """
    Permite al usuario seleccionar el nivel de creatividad del modelo.

    Se ofrecen varias opciones que corresponden a valores de temperatura.

    Returns:
        float: Valor de temperatura seleccionado para la generación de texto.
    """
    print("\nNivel de creatividad:")
    print("1. Muy preciso (0.2)")
    print("2. Equilibrado (0.5)")
    print("3. Creativo (0.8)")
    print("4. Muy creativo (1.0)")

    mapping = {
        1: 0.2,
        2: 0.5,
        3: 0.8,
        4: 1.0
    }

    while True:
        choice = input("Opción: ")

        try:
            value_int = int(choice)

            if value_int in mapping:
                temperature = mapping[value_int]
                print(f"Temperatura utilizada: {temperature}")
                return temperature
            else:
                print("Opción no válida. Elige entre 1 y 4.")

        except ValueError:
            print("El valor no es un número entero.")
    
    


def query_model(prompt, model_name, max_tokens, temperature):
    """
    Envía una consulta al modelo de lenguaje y devuelve la respuesta generada.

    Args:
        prompt (str): Texto de entrada del usuario.
        model_name (str): Nombre del modelo a utilizar.
        max_tokens (int | None): Límite máximo de tokens de salida.
        temperature (float): Nivel de creatividad del modelo.

    Returns:
        str: Respuesta generada por el modelo o mensaje de error en caso de fallo.
    """
    response = client.responses.create(
        model=model_name,
        input=prompt,
        max_output_tokens=max_tokens,
        temperature=temperature
    )
    
    start = time.perf_counter()
    end = time.perf_counter()

    latency = end - start

    print(f"Cantidad de tokens de input: {response.usage.input_tokens}")
    print(f"Cantidad de tokens de output: {response.usage.output_tokens}")
    print(f"tiempo de procesamiento (latencia) en segundos: {latency}")
       
    # -----------------------------
    # COSTE DEL OUTPUT
    # -----------------------------
    output_tokens = response.usage.output_tokens

    price_per_1m_output = output_pricing.get(model_name, 0)

    output_cost = (output_tokens / 1_000_000) * price_per_1m_output

    print(f"Coste del output: ${output_cost:.6f}")
    return response.output[0].content[0].text



# -----------------------------
# FLUJO PRINCIPAL

def main():
    """
    Función principal que ejecuta el flujo interactivo del programa.

    Permite al usuario:
    - Introducir una pregunta
    - Ver el número estimado de tokens
    - Seleccionar un modelo de lenguaje
    - Definir el límite de tokens de salida
    - Elegir el nivel de creatividad
    - Obtener una respuesta generada por el modelo

    El programa se ejecuta en bucle hasta que el usuario decide salir.
    """
    
    while True:
        # 1. Input usuario
        print("Oprime 1 para preguntar")
        print("Oprime 2 para salir")
        menu = input()

        if menu == "1":
            
            prompt = input("Introduce tu pregunta:\n")
            
            if not prompt.strip():
                print("La pregunta no puede estar vacía.\n")
                continue
            # 2. Tokens estimados
            tokens = estimate_tokens(prompt)
            print(f"\nTokens estimados: {tokens}")

            # 3. Selección modelo
            model = select_model(tokens)
                
            # 4. Max tokens
            max_tokens = select_max_tokens()

            # 5. Creatividad
            temperature = select_temperature()

            # 6. Llamada al modelo
            print("\nConsultando modelo...\n")

            answer = query_model(
                prompt,
                model["name"],
                max_tokens,
                temperature
            )

            print("Respuesta:\n")
            print(answer)

        elif menu == "2":
            print("¡Hasta pronto!")
            break
        
        else:
            print("Opción no válida")
        


if __name__ == "__main__":
    main()
