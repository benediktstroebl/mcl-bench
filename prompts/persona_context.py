def get_persona_context_prompt(persona, intent: str) -> str:
    """Generate the prompt for the persona context setting.
    
    Args:
        persona: The persona object containing user attributes
        intent: The user's original question/intent
        
    Returns:
        str: Formatted prompt with persona context
    """
    location = persona.geo.location_str()
    
    prompt = f"""Given the following details about the user, please respond to the question below:

USER INFORMATION:
- Age: {persona.age} years old
- Gender: {persona.sex}
- Location: {location}
  * City: {persona.geo.city}
  * State/Region: {persona.geo.state}
  * Country: {persona.geo.country}
- Political leaning: {persona.political_leaning}

QUESTION:
{intent}"""
    
    return prompt 