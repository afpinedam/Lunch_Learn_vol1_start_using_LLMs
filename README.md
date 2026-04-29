# LLM Chatbot Console Tool

Script en Python para interactuar con modelos de lenguaje (LLMs) desde consola, permitiendo estimar costes y personalizar respuestas.

---

## Descripción

El programa permite:

- Introducir preguntas por consola  
- Estimar tokens de entrada  
- Comparar costes entre modelos  
- Seleccionar modelo  
- Configurar límite de tokens  
- Ajustar creatividad (temperature)  
- Obtener respuesta del modelo  

---

## Técnica

- Estimación: **1 token ≈ 4 caracteres**
- Coste: calculado según precio por **1M tokens**
- Parámetros:
  - `max_tokens`: longitud de respuesta  
  - `temperature`: nivel de creatividad  

---

## Arquitectura
main()
├── estimate_tokens()
├── estimate_cost()
├── select_model()
├── select_max_tokens()
├── select_temperature()
└── query_model()

# Configuración

Agregar la key en el achivo .env:

# Modelos
Modelo de precio de input según la página de openAI (https://platform.openai.com/pricing):
gpt-4o-mini	$0.15 / 1M
gpt-4.1-mini	$0.40 / 1M
gpt-4.1	$2.00 / 1M
gpt-5-nano	$0.05 / 1M
gpt-5-mini	$0.25 / 1M